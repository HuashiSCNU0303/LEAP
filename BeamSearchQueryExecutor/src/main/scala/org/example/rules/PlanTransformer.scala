package org.example.rules

import org.apache.spark.sql.catalyst.catalog.HiveTableRelation
import org.apache.spark.sql.catalyst.expressions.{And, Attribute, Contains, EndsWith, Equality, Expression, GreaterThan, GreaterThanOrEqual, In, InSet, IsNotNull, IsNull, LessThan, LessThanOrEqual, Like, Literal, Not, Or, PredicateHelper, StartsWith}
import org.apache.spark.sql.catalyst.plans.{Inner, InnerLike}
import org.apache.spark.sql.catalyst.plans.logical.{Filter, Join, LogicalPlan}
import org.apache.spark.sql.execution.datasources.LogicalRelation
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.write
import org.json4s.{Formats, NoTypeHints}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer


// 和python的变量名规则对齐
case class ConditionNode(op_type: String, operator: String,
                         left_value: String, right_value: String)

case class PlanNode(node_type: String, condition: Seq[ConditionNode],
                    relation_name: String, table_cardinality: Long)


class PlanTransformer(exprIdMap: Map[Long, String]) extends PredicateHelper with DupEqualityHandler {

  implicit val formats: Formats = Serialization.formats(NoTypeHints)
  val planCache: mutable.Map[LogicalPlan, String] = mutable.Map[LogicalPlan, String]()

  private def transformCondition(condition: Expression): Seq[ConditionNode] = condition match {
    case And(cond1, cond2) =>
      val conditionNodes1 = transformCondition(cond1)
      val conditionNodes2 = transformCondition(cond2)
      Seq(ConditionNode("Bool", "AND", "", "")) ++ conditionNodes1 ++ conditionNodes2 ++ Seq(ConditionNode("Bool", ") AND", "", ""))

    case Or(cond1, cond2) =>
      val conditionNodes1 = transformCondition(cond1)
      val conditionNodes2 = transformCondition(cond2)
      Seq(ConditionNode("Bool", "OR", "", "")) ++ conditionNodes1 ++ conditionNodes2 ++ Seq(ConditionNode("Bool", ") OR", "", ""))

    case Not(cond) =>
      val conditionNodes = transformCondition(cond)
      Seq(ConditionNode("Bool", "NOT", "", "")) ++ conditionNodes ++ Seq(ConditionNode("Bool", ") NOT", "", ""))

    case _ =>
      transformSingleCondition(condition)
  }

  private def transformSingleCondition(condition: Expression): Seq[ConditionNode] = condition match {
    case Equality(ar: Attribute, l: Literal) =>
      Seq(ConditionNode("Compare", "=", exprIdMap(ar.exprId.id), l.value.toString))
    case Equality(l: Literal, ar: Attribute) =>
      Seq(ConditionNode("Compare", "=", exprIdMap(ar.exprId.id), l.value.toString))
    case Equality(ar1: Attribute, ar2: Attribute) =>
      Seq(ConditionNode("Compare", "=", exprIdMap(ar1.exprId.id), exprIdMap(ar2.exprId.id)))

    case op @ LessThan(ar: Attribute, l: Literal) =>
      Seq(ConditionNode("Compare", "<", exprIdMap(ar.exprId.id), l.value.toString))
    case op @ LessThan(l: Literal, ar: Attribute) =>
      Seq(ConditionNode("Compare", ">", exprIdMap(ar.exprId.id), l.value.toString))
    case op @ LessThan(ar1: Attribute, ar2: Attribute) =>
      Seq(ConditionNode("Compare", "<", exprIdMap(ar1.exprId.id), exprIdMap(ar2.exprId.id)))

    case op @ LessThanOrEqual(ar: Attribute, l: Literal) =>
      Seq(ConditionNode("Compare", "<=", exprIdMap(ar.exprId.id), l.value.toString))
    case op @ LessThanOrEqual(l: Literal, ar: Attribute) =>
      Seq(ConditionNode("Compare", ">=", exprIdMap(ar.exprId.id), l.value.toString))
    case op @ LessThanOrEqual(ar1: Attribute, ar2: Attribute) =>
      Seq(ConditionNode("Compare", "<=", exprIdMap(ar1.exprId.id), exprIdMap(ar2.exprId.id)))

    case op @ GreaterThan(ar: Attribute, l: Literal) =>
      Seq(ConditionNode("Compare", ">", exprIdMap(ar.exprId.id), l.value.toString))
    case op @ GreaterThan(l: Literal, ar: Attribute) =>
      Seq(ConditionNode("Compare", "<", exprIdMap(ar.exprId.id), l.value.toString))
    case op @ GreaterThan(ar1: Attribute, ar2: Attribute) =>
      Seq(ConditionNode("Compare", ">", exprIdMap(ar1.exprId.id), exprIdMap(ar2.exprId.id)))

    case op @ GreaterThanOrEqual(ar: Attribute, l: Literal) =>
      Seq(ConditionNode("Compare", ">=", exprIdMap(ar.exprId.id), l.value.toString))
    case op @ GreaterThanOrEqual(l: Literal, ar: Attribute) =>
      Seq(ConditionNode("Compare", "<=", exprIdMap(ar.exprId.id), l.value.toString))
    case op @ GreaterThanOrEqual(ar1: Attribute, ar2: Attribute) =>
      Seq(ConditionNode("Compare", ">=", exprIdMap(ar1.exprId.id), exprIdMap(ar2.exprId.id)))

    case In(ar: Attribute, expList) if expList.forall(e => e.isInstanceOf[Literal]) =>
      val hSet = expList.map(e => "\"" + e.eval().toString + "\"")  // 统一转字符串，记得加双引号！！！！！
      Seq(ConditionNode("Compare", "IN", exprIdMap(ar.exprId.id), hSet.mkString("|[,]|")))
    case InSet(ar: Attribute, set) =>
      val hSet = set.map(e => "\"" + e.toString + "\"")
      Seq(ConditionNode("Compare", "IN", exprIdMap(ar.exprId.id), hSet.mkString("|[,]|")))

    case IsNull(ar: Attribute) =>
      Seq(ConditionNode("Compare", "isnull", exprIdMap(ar.exprId.id), ""))

    case IsNotNull(ar: Attribute) =>
      Seq(ConditionNode("Compare", "isnotnull", exprIdMap(ar.exprId.id), ""))

    case Like(ar: Attribute, l: Literal, _) =>
      Seq(ConditionNode("Compare", "LIKE", exprIdMap(ar.exprId.id), l.value.toString))

    case Contains(ar: Attribute, l: Literal) =>
      Seq(ConditionNode("Compare", "Contains", exprIdMap(ar.exprId.id), l.value.toString))

    case StartsWith(ar: Attribute, l: Literal) =>
      Seq(ConditionNode("Compare", "StartsWith", exprIdMap(ar.exprId.id), l.value.toString))

    case EndsWith(ar: Attribute, l: Literal) =>
      Seq(ConditionNode("Compare", "EndsWith", exprIdMap(ar.exprId.id), l.value.toString))

    case _ => Seq()
  }

  def transformPlan(plan: LogicalPlan): Seq[PlanNode] = {
    def traverse(node: LogicalPlan): Seq[PlanNode] = {
      node match {
        case Join(left, right, joinType, condition, _) =>
          // 'Inner', 'LeftOuter', ...,去掉原始输出后面的"$"
          val nodeType = joinType.getClass.getSimpleName.dropRight(1)
          val joinCondition = transformCondition(condition.get)
          Seq(PlanNode(nodeType, joinCondition, "", -1)) ++
            traverse(left) ++ traverse(right) ++ 
            Seq(PlanNode(s") $nodeType", joinCondition, "", -1))
        case Filter(condition, child) =>
          val (relationName, tableCard) = child match {
            case LogicalRelation(_, _, catalogTable, _) =>
              (catalogTable.get.identifier.table, catalogTable.get.stats.get.rowCount.get.toLong)
            case HiveTableRelation(tableMeta, _, _, _, _) =>
              (tableMeta.identifier.table, tableMeta.stats.get.rowCount.get.toLong)
            case _ => ("", -1L)
          }
          Seq(PlanNode("Filter", transformCondition(condition), relationName, tableCard))
        case node => node.children.flatMap(child => traverse(child))
      }
    }
    traverse(plan)
  }

  def transformString(plan: LogicalPlan): String = {
    val string = if (planCache.contains(plan)) {
      planCache(plan)
    } else {
      val newPlan = removeDupEquality(plan)
      val jsonString: String = write(Map("seq" -> transformPlan(newPlan)))
      planCache(plan) = jsonString
      jsonString
    }
    // println(s"Generate plan string: $string")
    string
  }
}
