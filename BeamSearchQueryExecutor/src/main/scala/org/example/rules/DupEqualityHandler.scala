package org.example.rules

import org.apache.spark.sql.catalyst.expressions.{And, Attribute, EqualTo, Equality, ExpressionSet, PredicateHelper}
import org.apache.spark.sql.catalyst.plans.InnerLike
import org.apache.spark.sql.catalyst.plans.logical.{Join, LogicalPlan}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

trait DupEqualityHandler extends PredicateHelper {
  // 删除多余的condition，避免广播出现异常
  def removeDupEquality(plan: LogicalPlan): LogicalPlan = {
    val colGraph: mutable.Map[Long, mutable.Set[Long]] = mutable.Map.empty

    def addEdge(node1: Long, node2: Long): Unit = {
      colGraph.getOrElseUpdate(node1, mutable.Set.empty) += node2
      colGraph.getOrElseUpdate(node2, mutable.Set.empty) += node1
    }

    def isConnected(node_1: Long, node_2: Long): Boolean = {
      def dfs(current: Long, target: Long, visited: mutable.Set[Long]): Boolean = {
        if (current == target) true
        else if (!visited.contains(current)) {
          visited += current
          colGraph.getOrElse(current, Set.empty).exists(nbr => dfs(nbr, target, visited))
        } else false
      }
      dfs(node_1, node_2, mutable.Set[Long]())
    }

    plan transformUp {
      case join@Join(left, right, joinType, Some(cond), hint) =>
        joinType match {
          case _: InnerLike =>
            val conds = splitConjunctivePredicates(cond)
            val filteredConds = conds.filter {
              case Equality(leftKey: Attribute, rightKey: Attribute) =>
                if (!isConnected(leftKey.exprId.id, rightKey.exprId.id)) {
                  addEdge(leftKey.exprId.id, rightKey.exprId.id)
                  true
                } else false
              case _ => true
            }
            Join(left, right, joinType, filteredConds.reduceOption(And), hint)
          case _ =>
            colGraph.clear()
            join
        }
    }
  }
}
