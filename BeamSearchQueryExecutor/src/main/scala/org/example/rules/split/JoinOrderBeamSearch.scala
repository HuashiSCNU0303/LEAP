package org.example.rules.split

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.catalog.HiveTableRelation
import org.apache.spark.sql.catalyst.expressions.{And, Attribute, AttributeSet, ExpressionSet, PredicateHelper}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical.{Join, JoinHint, LogicalPlan, Project}
import org.apache.spark.sql.execution.datasources.LogicalRelation
import org.apache.spark.sql.internal.SQLConf
import org.example.Configs
import org.example.rules.PlanTransformer

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.util.Random


abstract class JoinOrderBeamSearch(conf: SQLConf,
                                   items: Seq[LogicalPlan],
                                   conditions: ExpressionSet,
                                   output: Seq[Attribute]) extends PredicateHelper with Logging {

  var estCount = 0
  val joinItems: Seq[JoinPlan] = items.zipWithIndex.map {
    case (item, id) => JoinPlan(Set(id), item, ExpressionSet())
  }
  val attrIdToItem: Map[Long, Int] = items.zipWithIndex.flatMap {
    case (item, id) => item.flatMap {
      case rel: LogicalRelation => rel.output.map(at => at.exprId.id -> id)
      case rel: HiveTableRelation => rel.output.map(at => at.exprId.id -> id)
      case _ => Map()
    }
  }.toMap
  val joinGraph: Map[Int, mutable.Set[Int]] = {
    val graph = joinItems.indices.map(id => id -> mutable.Set[Int]()).toMap
    conditions.foreach { cond =>
      val referencedTables = cond.references.map(at => attrIdToItem(at.exprId.id)).toSet
      if (referencedTables.size == 2) {
        val tables = referencedTables.toSeq
        graph(tables.head) += tables(1)
        graph(tables(1)) += tables.head
      }
    }
    println(s"graph = $graph")
    graph
  }
  val planTransformer: PlanTransformer = {
    new PlanTransformer(items.flatMap(_.collect {
      case rel: HiveTableRelation => rel.output.map(at => at.exprId.id -> s"${rel.tableMeta.identifier.table}.${at.name}")
      case rel: LogicalRelation => rel.output.map(at => at.exprId.id -> s"${rel.catalogTable.get.identifier.table}.${at.name}")
    }).flatten.toMap)
  }

  def search(): LogicalPlan = {
    val topOutputSet = AttributeSet(output)
    val bestPlan = findBestPlan(conf, conditions, topOutputSet)
    bestPlan.plan match {
      case p@Project(projectList, j: Join) if projectList != output =>
        assert(topOutputSet == p.outputSet)
        p.copy(projectList = output)
      case finalPlan if !sameOutput(finalPlan, output) =>
        Project(output, finalPlan)
      case finalPlan =>
        finalPlan
    }
  }

  @scala.annotation.tailrec
  final def findTopK(plans: ListBuffer[JoinPlan], low: Int, high: Int, k: Int): Unit = {

    def partition(plans: ListBuffer[JoinPlan], low: Int, high: Int): Int = {
      val pivot = plans(high)

      val plansForComparison = for (j <- low until high) yield plans(j).plan
      val comparisons = Comparator.compareCost(plansForComparison, pivot.plan, planTransformer)

      var i = low
      var j = high - 1
      while (i <= j) {
        if (comparisons(i - low)) {
          i += 1
        } else if (!comparisons(j - low)) {
          j -= 1
        } else {
          val temp = plans(i)
          plans(i) = plans(j)
          plans(j) = temp
          i += 1
          j -= 1
        }
      }
      val temp = plans(i)
      plans(i) = pivot
      plans(high) = temp
      i
    }

    if (low < high) {
      val pi = partition(plans, low, high)
      println(s"low = $low, high = $high, after partition(), to $pi")
      // 前半部分（含pi自己）的个数为pi - low + 1，如果 > k，那么在前半部分找第k个 pi - low + 1 > k
      if (pi > low + k - 1) {
        findTopK(plans, low, pi - 1, k)
      }
      // 如果 < k，那么在后半部分找第k - (pi - low + 1)个
      else if (pi < low + k - 1) {
        findTopK(plans, pi + 1, high, k - (pi - low + 1))
      }
    }
  }

  def findTopKCost(plans: ListBuffer[JoinPlan], k: Int): ListBuffer[JoinPlan] = {
    if (plans.size <= k) {
      return plans
    }
    val indices = Comparator.findTopKPlans(for (plan <- plans) yield plan.plan, planTransformer, k)
    println(s"indices = $indices, ${indices.map(index => index.toInt)}")
    ListBuffer() ++= indices.map(index => plans(index.toInt))
  }

  def compareCost(plans: Seq[JoinPlan], pivot: Option[JoinPlan]): Seq[Boolean] = {
    if (pivot.isEmpty) {
      return plans.map(_ => true)
    }
    Comparator.compareCost(plans.map(_.plan), pivot.get.plan, planTransformer)
  }

  def findBestPlan(conf: SQLConf, conditions: ExpressionSet, output: AttributeSet): JoinPlan

  def isConnectedJoin(items: List[Int]): Boolean = {
    def traverse(node: Int, visited: mutable.Set[Int]): Unit = {
      if (!visited(node) && items.contains(node)) {
        visited += node
        joinGraph(node).foreach(n => traverse(n, visited))
      }
    }

    // Check if the graph is connected
    val visited = mutable.Set[Int]()
    traverse(items.head, visited)
    visited.size == items.size
  }

  def sameOutput(plan: LogicalPlan, expectedOutput: Seq[Attribute]): Boolean = {
    val thisOutput = plan.output
    thisOutput.length == expectedOutput.length && thisOutput.zip(expectedOutput).forall {
      case (a1, a2) => a1.semanticEquals(a2)
    }
  }

  def buildJoin(oneJoinPlan: JoinPlan,
                otherJoinPlan: JoinPlan,
                conditions: ExpressionSet,
                topOutput: AttributeSet,
                joinHint: JoinHint = JoinHint.NONE): Option[JoinPlan] = {
    if (oneJoinPlan.itemIds.intersect(otherJoinPlan.itemIds).nonEmpty) {
      // Should not join two overlapping item sets.
      return None
    }

    // 已经在前面做了判断，这里不会有笛卡尔积
    val onePlan = oneJoinPlan.plan
    val otherPlan = otherJoinPlan.plan
    val joinConds = conditions
      .filterNot(l => canEvaluate(l, onePlan))
      .filterNot(r => canEvaluate(r, otherPlan))
      .filter(e => e.references.subsetOf(onePlan.outputSet ++ otherPlan.outputSet))
    val (left, right) = (onePlan, otherPlan)

    val newJoin = Join(left, right, Inner, joinConds.reduceOption(And), joinHint)
    val collectedJoinConds = joinConds ++ oneJoinPlan.joinConds ++ otherJoinPlan.joinConds
    val remainingConds = conditions -- collectedJoinConds
    // 看起来这个报错不需要管的样子
    val neededAttr = AttributeSet(remainingConds.flatMap(_.references)) ++ topOutput
    val neededFromNewJoin = newJoin.output.filter(neededAttr.contains)
    val newPlan =
      if ((newJoin.outputSet -- neededFromNewJoin).nonEmpty) {
        Project(neededFromNewJoin, newJoin)
      } else {
        newJoin
      }

    val itemIds = oneJoinPlan.itemIds.union(otherJoinPlan.itemIds)
    Some(JoinPlan(itemIds, newPlan, collectedJoinConds))
  }

  case class JoinPlan(var itemIds: Set[Int],
                      plan: LogicalPlan,
                      joinConds: ExpressionSet)

}
