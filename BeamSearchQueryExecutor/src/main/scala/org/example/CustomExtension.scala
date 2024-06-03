package org.example

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSessionExtensions
import org.example.rules.split.{Comparator, HintJoinSelection, SplitOrLeftDeepJoinReorder}

class CustomExtension extends (SparkSessionExtensions => Unit) with Logging {

  override def apply(extensions: SparkSessionExtensions): Unit = {
    Configs.getReorderClassName match {
      case "SplitOrLeftDeepJoinReorder" =>
        Comparator.init()
        extensions.injectPreCBORule(SplitOrLeftDeepJoinReorder)
        extensions.injectPlannerStrategy(HintJoinSelection)
      case _ => logWarning("Unknown join reorder class name, skip injecting join reorder rule...")
    }
  }
}
