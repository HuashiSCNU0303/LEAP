package org.example.rules.split

import comparator.{BroadcastRequest, ComparatorServiceGrpc, CompareRequest, TopKRequest}
import io.grpc.ManagedChannel
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.catalyst.plans.logical.statsEstimation.EstimationUtils
import org.apache.spark.sql.internal.SQLConf
import org.example.Configs
import org.example.rules.PlanTransformer

object Comparator {

  var channel: ManagedChannel = _
  var stub: ComparatorServiceGrpc.ComparatorServiceBlockingStub = _

  def init(): Unit = {
    channel = NettyChannelBuilder.forAddress(Configs.getRPCHost, Configs.getRPCPort).usePlaintext().build()
    stub = ComparatorServiceGrpc.blockingStub(channel)

    val request = CompareRequest(plans = Seq[String]("test"))
    stub.compareCost(request)
  }

  def compareCost(plans: Seq[LogicalPlan],
                  pivot: LogicalPlan,
                  planTransformer: PlanTransformer): Seq[Boolean] = {
    val planStrings = for (plan <- plans) yield planTransformer.transformString(plan)
    val startTime = System.nanoTime()
    val request = CompareRequest(planStrings, planTransformer.transformString(pivot))
    val result = stub.compareCost(request).result
    println(s"compareCost() time cost = ${(System.nanoTime() - startTime) / 1e6} ms")
    result
  }

  def canBeBroadcast(plan: LogicalPlan,
                     conf: SQLConf,
                     planTransformer: PlanTransformer): Boolean = {
    val thresholds = conf.autoBroadcastJoinThreshold / EstimationUtils.getSizePerRow(plan.output, plan.stats.attributeStats).toLong
    val planStrings = planTransformer.transformString(plan)
    val request = BroadcastRequest(Seq(planStrings), Seq(thresholds))
    // println(s"canBeBroadcast(), plan = $plan, threshold = $thresholds")
    stub.canBeBroadcast(request).result.head
  }

  def canBeBroadcast(plans: Seq[LogicalPlan],
                     conf: SQLConf,
                     planTransformer: PlanTransformer): Seq[Boolean] = {
    val thresholds = for (plan <- plans) yield conf.autoBroadcastJoinThreshold / EstimationUtils.getSizePerRow(plan.output, plan.stats.attributeStats).toLong
    val planStrings = for (plan <- plans) yield planTransformer.transformString(plan)
    val startTime = System.nanoTime()
    val request = BroadcastRequest(planStrings, thresholds)
    val result = stub.canBeBroadcast(request).result
    println(s"canBeBroadcast() time cost = ${(System.nanoTime() - startTime) / 1e6} ms")
    result
  }

  def findTopKPlans(plans: Seq[LogicalPlan],
                    planTransformer: PlanTransformer, k: Int): Seq[Long] = {
    val planStrings = for (plan <- plans) yield planTransformer.transformString(plan)
    val startTime = System.nanoTime()
    val request = TopKRequest(planStrings, k)
    val result = stub.getTopKPlans(request).result
    println(s"findTopKPlans() time cost = ${(System.nanoTime() - startTime) / 1e6} ms")
    result
  }
}

