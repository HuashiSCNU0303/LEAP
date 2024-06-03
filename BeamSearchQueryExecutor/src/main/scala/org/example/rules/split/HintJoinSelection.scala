package org.example.rules.split

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.catalog.HiveTableRelation
import org.apache.spark.sql.catalyst.plans.InnerLike
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, BROADCAST, HintInfo, Join, JoinHint, LogicalPlan, Project, ReturnAnswer, Statistics}
import org.apache.spark.sql.execution.adaptive.LogicalQueryStage
import org.apache.spark.sql.execution.datasources.LogicalRelation
import org.apache.spark.sql.execution.{SparkPlan, SparkStrategy}
import org.apache.spark.sql.internal.SQLConf
import org.example.rules.PlanTransformer


case class HintJoinSelection(spark: SparkSession) extends SparkStrategy with Logging {

  private def getHint(plan: LogicalPlan, conf: SQLConf, planTransformer: PlanTransformer): Option[HintInfo] = {
    if (canBeCompared(plan) && Comparator.canBeBroadcast(plan, conf, planTransformer)) {
      Some(HintInfo(strategy = Some(BROADCAST)))
    } else None
  }

  private def canBeCompared(plan: LogicalPlan): Boolean = {
    plan.find {
      case node: Aggregate => true
      case _ => false
    }.isEmpty
  }

  override def apply(plan: LogicalPlan): Seq[SparkPlan] = {
    plan match {
      // 生成sparkPlan时，基于ReturnAnswer(plan)进行，因此猜测所有的待处理逻辑计划的根节点均为ReturnAnswer
      case ReturnAnswer(rootPlan) =>

        val startTime = System.nanoTime()
        val conf = spark.sessionState.conf
        // 此时说明有AQE过来了
        if (rootPlan.collectFirst { case j: LogicalQueryStage => j }.isDefined) {
          return Nil
        }
        // 之前加过Hint了
        if (rootPlan.collectFirst { case j@Join(_, _, _, _, hint) if hint != JoinHint.NONE => j }.isDefined) {
          return Nil
        }

        val planTransformer: PlanTransformer = new PlanTransformer(rootPlan.collect {
          case rel: HiveTableRelation => rel.output.map(at => at.exprId.id -> s"${rel.tableMeta.identifier.table}.${at.name}")
          case rel: LogicalRelation => rel.output.map(at => at.exprId.id -> s"${rel.catalogTable.get.identifier.table}.${at.name}")
        }.flatten.toMap)

        val result = rootPlan transform {
          case j@Join(left, right, joinType: InnerLike, cond, JoinHint.NONE) if canBeCompared(j) =>
            val isBushy = right.find {
              case _: Join => true
              case _ => false
            }.isDefined
            val (leftHint, rightHint) = if (isBushy) {
              (Some(HintInfo(strategy = Some(BROADCAST))), None)
            } else {
              // 没有HINT的时候再加HINT
              val rightHint = getHint(right, conf, planTransformer)
              // 右边不能广播，再考虑左边
              val leftHint = if (rightHint.isEmpty || rightHint.get.strategy.get != BROADCAST) {
                getHint(left, conf, planTransformer)
              } else None
              (leftHint, rightHint)
            }
            Join(left, right, joinType, cond, JoinHint(leftHint = leftHint, rightHint = rightHint))
        }

        logWarning(s"In Join Selection, revised plan = ${result.treeString}, time cost = ${(System.nanoTime() - startTime) / 1e6} ms")
        planLater(result) :: Nil

      case _ => Nil
    }
  }
}
