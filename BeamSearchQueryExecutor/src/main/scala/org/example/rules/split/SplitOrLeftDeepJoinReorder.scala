package org.example.rules.split

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.catalog.HiveTableRelation
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeSet, ExpressionSet}
import org.apache.spark.sql.catalyst.plans.InnerLike
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.internal.SQLConf
import org.example.rules.{DupEqualityHandler, JoinOrderGenerator}
import org.apache.spark.sql.catalyst.optimizer.{JoinReorderDP, OrderedJoin}
import org.apache.spark.sql.execution.datasources.LogicalRelation
import org.example.Configs

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.util.control.Breaks._

case class SplitOrLeftDeepJoinReorder(spark: SparkSession) extends Rule[LogicalPlan] with Logging {
  override def apply(plan: LogicalPlan): LogicalPlan = {
    val startTime = System.nanoTime()
    val newPlan = SplitOrLeftDeepJoinOrderGenerator(plan)

    logWarning(s"SplitOrLeftDeepJoinReorder Join Reorder time cost = ${(System.nanoTime() - startTime) / 1e6} ms")
    newPlan
  }
}

private object SplitOrLeftDeepJoinOrderGenerator extends JoinOrderGenerator {

  def apply(plan: LogicalPlan): LogicalPlan = {
    val result = plan transformDown {
      case j@Join(left, right, joinType: InnerLike, Some(cond), JoinHint.NONE) =>
        logWarning(s"join, joinType = $joinType, cond = $cond")
        val reorderedNode = reorder(j, j.output)
        reorderedNode
      case p@Project(projectList, Join(left, right, joinType: InnerLike, Some(cond), JoinHint.NONE))
        if projectList.forall(_.isInstanceOf[Attribute]) =>
        logWarning(s"project, joinType = $joinType, cond = $cond")
        val reorderedNode = reorder(p, p.output)
        reorderedNode
    }
    val transformedResult = result transform {
      case OrderedJoin(left, right, jt, cond) => Join(left, right, jt, cond, JoinHint.NONE)
    }
    println(s"finalPlan: \n${transformedResult.toString}")
    transformedResult
  }

  private def reorder(plan: LogicalPlan, output: Seq[Attribute]): LogicalPlan = {
    val (items, conditions) = extractInnerJoins(plan)
    val result =
      if (items.size > 2 && items.size <= conf.joinReorderDPThreshold && conditions.nonEmpty) {
        logWarning(s"start join reorder")
        if (items.forall(item => item.find {
          case Aggregate(_, _, _) => true
          case _ => false
        }.isEmpty)) {
          logWarning(s"don't contain Aggregate, start beam search")
          SplitOrLeftDeepJoinOrderBeamSearch(conf, items, conditions, output)
        } else {
          logWarning(s"contains Aggregate, start CBO join reorder")
          JoinReorderDP.search(conf, items, conditions, output)
        }
      } else {
        logWarning(s"skip join reorder")
        plan
      }
    replaceWithOrderedJoin(result)
  }
}


private class SplitOrLeftDeepJoinOrderBeamSearch(conf: SQLConf,
                                                 items: Seq[LogicalPlan],
                                                 conditions: ExpressionSet,
                                                 output: Seq[Attribute])
  extends JoinOrderBeamSearch(conf, items, conditions, output) {

  private lazy val smallTableId: Set[Int] = {
    val comparisons = Comparator.canBeBroadcast(joinItems.map(_.plan), conf, planTransformer)
    joinItems.indices.filter(id => {
      if (comparisons(id)) {
        println(s"Detected small table: ${extractItemName(joinItems(id).plan).getOrElse("<unknown>")}")
      }
      comparisons(id)
    }).toSet
  }

  private def extractItemName(plan: LogicalPlan): Option[String] = {
    val tableNames = plan.collect {
      case l: LogicalRelation => l.catalogTable.get.identifier.table
      case h: HiveTableRelation => h.tableMeta.identifier.table
    }
    if (tableNames.size != 1) {
      None
    } else {
      Some(tableNames.head)
    }
  }

  private def beamSearchInEachLevel(plansInPreLevel: ListBuffer[JoinPlan],
                                    items: Seq[JoinPlan],
                                    levelId: Int,
                                    conditions: ExpressionSet,
                                    output: AttributeSet): ListBuffer[JoinPlan] = {
    val plansInNewLevel = ListBuffer[JoinPlan]()
    for (plan <- plansInPreLevel) {
      for (item <- items) {
        if (isConnectedJoin((plan.itemIds ++ item.itemIds).toList)) {
          val newPlan = buildJoin(plan, item, conditions, output)
          if (newPlan.isDefined) plansInNewLevel += newPlan.get
        }
      }
    }

    println(s"In level $levelId, there are ${plansInNewLevel.size} plans, shrink to ${Configs.getBeamSearchTopK} plans")
    findTopKCost(plansInNewLevel, Configs.getBeamSearchTopK)
  }

  private def genItemPairs(items: Seq[JoinPlan], conditions: ExpressionSet, output: AttributeSet): ListBuffer[JoinPlan] = {
    var itemPairs: ListBuffer[JoinPlan] = new ListBuffer[JoinPlan]
    val newItems = items.sortBy(_.itemIds.head)
    for (id_1 <- newItems.indices) {
      for (id_2 <- id_1 + 1 until newItems.size) {
        if (isConnectedJoin((newItems(id_1).itemIds ++ newItems(id_2).itemIds).toList)) {
          val plan = buildJoin(newItems(id_1), newItems(id_2), conditions, output)
          if (plan.isDefined) itemPairs += plan.get
        }
      }
    }

    val filteredItemPairs = itemPairs.filter(plan => smallTableId.intersect(plan.itemIds).size == 1)
    if (filteredItemPairs.nonEmpty) {
      itemPairs = filteredItemPairs
    } else {
      println(s"In level 2, only ${filteredItemPairs.size} pairs have small table, skip filter")
    }
    println(s"In level 2, there are ${itemPairs.size} plans, shrink to ${Configs.getBeamSearchTopK} plans")
    findTopKCost(itemPairs, Configs.getBeamSearchTopK)
  }


  private def findSubPlans(candidatePlans: ListBuffer[JoinPlan], level: Int, items: Seq[JoinPlan]): Seq[JoinPlan] = {
    val comparison = Comparator.canBeBroadcast(candidatePlans.map(_.plan), conf, planTransformer)
    val maybeBroadcast = comparison.indices.filter(index => comparison(index)).map(index => candidatePlans(index))
    if (maybeBroadcast.nonEmpty) {
      println(s"${maybeBroadcast.size} sub plans can be broadcast")
      val itemIds = items.flatMap(item => item.itemIds).toSet
      val filteredPlans = maybeBroadcast.filter {
        candidate => {
          val otherItems = itemIds -- candidate.itemIds
          val numOtherSmallTables = otherItems.count(id => smallTableId.contains(id))
          val smallTablesQualified = numOtherSmallTables > 0 || otherItems.size <= 1
          isConnectedJoin(otherItems.toList) && smallTablesQualified
        }
      }
      if (filteredPlans.isEmpty) {
        println("No sub plan can be broadcast, due to disconnected joins or too few small tables")
      }
      filteredPlans
    }
    else {
      println(s"In level ${level + 1}, no plan can be broadcast, continue left-deep search")
      Seq[JoinPlan]()
    }
  }

  @Override
  private def findSubBestPlan_(items: Seq[JoinPlan],
                               conf: SQLConf,
                               conditions: ExpressionSet,
                               output: AttributeSet): JoinPlan = {
    if (items.size == 1) {
      items.head
    } else if (items.size == 2) {
      buildJoin(items.head, items.last, conditions, output).get
    } else {
      var curBestPlan: Option[JoinPlan] = None
      val candidatePlans = mutable.Buffer[ListBuffer[JoinPlan]](genItemPairs(items, conditions, output))

      breakable {
        for (level <- 2 until items.size) {
          val selectedSubPlans = findSubPlans(candidatePlans.last, level, items)
          if (selectedSubPlans.nonEmpty) {
            val candidateFullPlans = selectedSubPlans.map(subPlan => {
              val otherItemIds = items.flatMap(item => item.itemIds).toSet -- subPlan.itemIds
              val otherItems = otherItemIds.map(id => joinItems(id)).toSeq
              val otherPlan = findSubBestPlan_(otherItems, conf, conditions, output)
              val candidate = buildJoin(subPlan, otherPlan, conditions, output).get
              candidate.itemIds = subPlan.itemIds
              candidate
            }).toBuffer

            val bestPlanBySplit = findTopKCost(ListBuffer() ++= candidateFullPlans, 1).head
            println(s"Get ${candidateFullPlans.size} candidate split, find the best plan:\n ${bestPlanBySplit.plan}")
            candidatePlans(candidatePlans.size - 1) = ListBuffer[JoinPlan](selectedSubPlans.find(plan => plan.itemIds == bestPlanBySplit.itemIds).get)
            bestPlanBySplit.itemIds = items.flatMap(item => item.itemIds).toSet

            if (compareCost(Seq[JoinPlan](bestPlanBySplit), curBestPlan).head) {
              println(s"curBestPlan update from ${curBestPlan} to ${bestPlanBySplit}")
              curBestPlan = Some(bestPlanBySplit)
            }
          }

          val plansInNewLevel = beamSearchInEachLevel(candidatePlans.last, items, level + 1, conditions, output)
          // 剪枝
          val pruningComparison = compareCost(plansInNewLevel, curBestPlan)
          val filteredPlans = plansInNewLevel.zip(pruningComparison).collect {
            case (plan, comparison) if comparison => plan
          }
          candidatePlans += filteredPlans
          println(s"In level ${level + 1}, generate the following sub-plans:")
          filteredPlans.foreach(println)
          if (filteredPlans.isEmpty) {
            break
          }
        }
      }
      val allCandidatePlans: ListBuffer[JoinPlan] = ListBuffer[JoinPlan]()
      if (candidatePlans.last.nonEmpty) {
        println(s"For items = ${items.flatMap(_.itemIds)}, select best left-deep plan from ${candidatePlans.last.size} plans")
        allCandidatePlans += findTopKCost(candidatePlans.last, 1).head
      }
      if (curBestPlan.nonEmpty) {
        allCandidatePlans += curBestPlan.get
      }
      println(s"For items = ${items.flatMap(_.itemIds)}, select final best plan from ${allCandidatePlans.size} plans")
      findTopKCost(allCandidatePlans, 1).head
    }
  }

  override def findBestPlan(conf: SQLConf,
                            conditions: ExpressionSet,
                            output: AttributeSet): JoinPlan = findSubBestPlan_(joinItems, conf, conditions, output)
}

object SplitOrLeftDeepJoinOrderBeamSearch extends DupEqualityHandler {
  def apply(conf: SQLConf,
            items: Seq[LogicalPlan],
            conditions: ExpressionSet,
            output: Seq[Attribute]): LogicalPlan = {
    val instance = new SplitOrLeftDeepJoinOrderBeamSearch(conf, items, conditions, output)
    removeDupEquality(instance.search())
  }
}
