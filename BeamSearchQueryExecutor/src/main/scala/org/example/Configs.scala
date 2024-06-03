package org.example

import org.apache.spark.{SparkConf, SparkContext}

object Configs {
  val conf: SparkConf = SparkContext.getOrCreate().getConf

  def getRPCHost: String = conf.get("spark.card.rpc.host", "localhost")
  def getRPCPort: Int = conf.get("spark.card.rpc.port", "9009").toInt
  def getReorderClassName: String = conf.get("spark.card.join.reorder.class", "")
  def getBeamSearchTopK: Int = conf.get("spark.card.beamsearch.k", "4").toInt
}
