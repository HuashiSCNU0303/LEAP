package org.example.rules

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.{Attribute, ExpressionSet, PredicateHelper}
import org.apache.spark.sql.catalyst.optimizer.OrderedJoin
import org.apache.spark.sql.catalyst.plans.InnerLike
import org.apache.spark.sql.catalyst.plans.logical.{Join, JoinHint, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule

trait JoinOrderGenerator extends Rule[LogicalPlan] with Logging with PredicateHelper {

  def extractInnerJoins(plan: LogicalPlan): (Seq[LogicalPlan], ExpressionSet) = {
    plan match {
      case Join(left, right, _: InnerLike, Some(cond), JoinHint.NONE) =>
        val (leftPlans, leftConditions) = extractInnerJoins(left)
        val (rightPlans, rightConditions) = extractInnerJoins(right)
        (leftPlans ++ rightPlans, leftConditions ++ rightConditions ++
          splitConjunctivePredicates(cond))
      case Project(projectList, j @ Join(_, _, _: InnerLike, Some(cond), JoinHint.NONE))
        if projectList.forall(_.isInstanceOf[Attribute]) =>
        extractInnerJoins(j)
      case _ =>
        (Seq(plan), ExpressionSet())
    }
  }

  def replaceWithOrderedJoin(plan: LogicalPlan): LogicalPlan = plan match {
    case j @ Join(left, right, jt: InnerLike, Some(cond), JoinHint.NONE) =>
      val replacedLeft = replaceWithOrderedJoin(left)
      val replacedRight = replaceWithOrderedJoin(right)
      OrderedJoin(replacedLeft, replacedRight, jt, Some(cond))
    case p @ Project(projectList, j @ Join(_, _, _: InnerLike, Some(cond), JoinHint.NONE)) =>
      p.copy(child = replaceWithOrderedJoin(j))
    case _ =>
      plan
  }
}
