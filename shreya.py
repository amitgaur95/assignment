{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPDMjcUlW902OclSrO5zjgD",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/shreyavidhi/assignment/blob/main/shreya.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from pyspark.sql import SparkSession\n",
        "from pyspark.ml.feature import VectorAssembler, StringIndexer\n",
        "from pyspark.ml.classification import DecisionTreeClassifier\n",
        "from pyspark.ml.evaluation import BinaryClassificationEvaluator\n",
        "\n",
        "spark = SparkSession.builder.appName(\"GooglePlayStoreClassification\").getOrCreate()\n",
        "\n",
        "data = spark.read.csv('/googleplaystore_user_reviews.csv', header=True, inferSchema=True)\n",
        "\n",
        "data = data.dropna()\n",
        "\n",
        "selected_features = [\"App\", \"Translated_Review\", \"Sentiment\", \"Sentiment_Polarity\", \"Sentiment_Subjectivity\"]\n",
        "target_variable = \"Translated_Review\"\n",
        "\n",
        "app_indexer = StringIndexer(inputCol=\"App\", outputCol=\"AppIndex\")\n",
        "translated_review_indexer = StringIndexer(inputCol=\"Translated_Review\", outputCol=\"TranslatedReviewIndex\")\n",
        "sentiment_indexer = StringIndexer(inputCol=\"Sentiment\", outputCol=\"SentimentIndex\")\n",
        "\n",
        "data = app_indexer.fit(data).transform(data)\n",
        "data = translated_review_indexer.fit(data).transform(data)\n",
        "data = sentiment_indexer.fit(data).transform(data)\n",
        "data = data.withColumn(\"Sentiment_Polarity\", data[\"Sentiment_Polarity\"].cast(\"float\"))\n",
        "data = data.withColumn(\"Sentiment_Subjectivity\", data[\"Sentiment_Subjectivity\"].cast(\"float\"))\n",
        "\n",
        "vector_assembler = VectorAssembler(inputCols=[\"AppIndex\", \"TranslatedReviewIndex\", \"SentimentIndex\", \"Sentiment_Polarity\", \"Sentiment_Subjectivity\"],\n",
        "                                   outputCol=\"features\", handleInvalid=\"keep\")\n",
        "\n",
        "data = vector_assembler.transform(data)\n",
        "\n",
        "data = data.withColumn(\"Translated_Review\", data[\"Translated_Review\"].cast(\"float\"))\n",
        "(training_data, testing_data) = data.randomSplit([0.8, 0.2], seed=42)\n",
        "dt_model = DecisionTreeClassifier(featuresCol=\"features\", labelCol=target_variable)"
      ],
      "metadata": {
        "id": "40hlL_HCbanz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from pyspark.sql import SparkSession\n",
        "from pyspark.ml.feature import VectorAssembler, StringIndexer\n",
        "from pyspark.ml.classification import DecisionTreeClassifier\n",
        "from pyspark.ml.evaluation import BinaryClassificationEvaluator\n",
        "import pyspark.sql.functions as F\n",
        "\n",
        "# Create a Spark session\n",
        "spark = SparkSession.builder.appName(\"GooglePlayStoreClassification\").getOrCreate()\n",
        "\n",
        "# Load the dataset (replace 'path/to/your/dataset' with the actual path)\n",
        "data = spark.read.csv('/googleplaystore.csv', header=True, inferSchema=True)\n",
        "\n",
        "# Drop rows with missing values for simplicity in this example\n",
        "data = data.dropna()\n",
        "\n",
        "# Select relevant features and target variable\n",
        "selected_features = [\"Rating\", \"Reviews\", \"Price\", \"Category\", \"Content Rating\"]\n",
        "target_variable = \"Rating\"  # Assuming 'Type' is the target variable (free or paid)\n",
        "\n",
        "# Convert categorical variables to numerical using StringIndexer\n",
        "category_indexer = StringIndexer(inputCol=\"Category\", outputCol=\"CategoryIndex\")\n",
        "content_rating_indexer = StringIndexer(inputCol=\"Content Rating\", outputCol=\"ContentRatingIndex\")\n",
        "\n",
        "data = category_indexer.fit(data).transform(data)\n",
        "data = content_rating_indexer.fit(data).transform(data)\n",
        "\n",
        "data = data.withColumn(\"Rating\", data[\"Rating\"].cast(\"float\"))\n",
        "data = data.withColumn(\"Reviews\", data[\"Reviews\"].cast(\"float\"))\n",
        "data = data.withColumn(\"Price\", data[\"Price\"].cast(\"float\"))\n",
        "\n",
        "# Create a feature vector\n",
        "vector_assembler = VectorAssembler(inputCols=[\"Rating\", \"Reviews\", \"Price\", \"CategoryIndex\", \"ContentRatingIndex\", \"Rating\"],\n",
        "                                   outputCol=\"features\", handleInvalid=\"keep\")\n",
        "data = vector_assembler.transform(data)\n",
        "\n",
        "# Split the data into training and testing sets\n",
        "(training_data, testing_data) = data.randomSplit([0.8, 0.2], seed=42)\n",
        "\n",
        "# Create a DecisionTreeClassifier model\n",
        "dt_model = DecisionTreeClassifier(featuresCol=\"features\", labelCol=target_variable)\n",
        "\n",
        "data = data.filter(F.col(\"Rating\").isNotNull())\n",
        "data = data.filter(F.col(\"Reviews\").isNotNull())\n",
        "data = data.filter(F.col(\"Price\").isNotNull())\n",
        "\n",
        "# # Train the model\n",
        "dt_model = dt_model.fit(training_data)\n",
        "\n",
        "# # Make predictions on the testing set\n",
        "# predictions = dt_model.transform(testing_data)\n",
        "\n",
        "# # Evaluate the model\n",
        "# evaluator = BinaryClassificationEvaluator(labelCol=target_variable, metricName=\"areaUnderROC\")\n",
        "# auc = evaluator.evaluate(predictions)\n",
        "# print(\"Area under the ROC curve (AUC) on test data =\", auc)\n",
        "\n",
        "# # You can now use the trained model to make predictions on new data\n",
        "# # For example, if you have a new dataset 'new_data', you can use:\n",
        "# # new_predictions = dt_model.transform(vector_assembler.transform(new_data))\n",
        "\n",
        "# # Stop the Spark session\n",
        "# spark.stop()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "m7asOYmWpjCp",
        "outputId": "ede49d5d-fb73-4015-82b0-9bf87b58302f"
      },
      "execution_count": 150,
      "outputs": [
        {
          "output_type": "error",
          "ename": "Py4JJavaError",
          "evalue": "An error occurred while calling o10841.fit.\n: org.apache.spark.SparkException: Job aborted due to stage failure: Task 0 in stage 671.0 failed 1 times, most recent failure: Lost task 0.0 in stage 671.0 (TID 693) (f139c0dcbf82 executor driver): java.lang.RuntimeException: Labels MUST be Integers, but got 4.5\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.hashAgg_doAggregate_max_0$(Unknown Source)\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.hashAgg_doConsume_0$(Unknown Source)\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.hashAgg_doAggregateWithoutKey_0$(Unknown Source)\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.processNext(Unknown Source)\n\tat org.apache.spark.sql.execution.BufferedRowIterator.hasNext(BufferedRowIterator.java:43)\n\tat org.apache.spark.sql.execution.WholeStageCodegenEvaluatorFactory$WholeStageCodegenPartitionEvaluator$$anon$1.hasNext(WholeStageCodegenEvaluatorFactory.scala:43)\n\tat scala.collection.Iterator$$anon$10.hasNext(Iterator.scala:460)\n\tat org.apache.spark.shuffle.sort.BypassMergeSortShuffleWriter.write(BypassMergeSortShuffleWriter.java:140)\n\tat org.apache.spark.shuffle.ShuffleWriteProcessor.write(ShuffleWriteProcessor.scala:59)\n\tat org.apache.spark.scheduler.ShuffleMapTask.runTask(ShuffleMapTask.scala:104)\n\tat org.apache.spark.scheduler.ShuffleMapTask.runTask(ShuffleMapTask.scala:54)\n\tat org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:161)\n\tat org.apache.spark.scheduler.Task.run(Task.scala:141)\n\tat org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$4(Executor.scala:620)\n\tat org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:64)\n\tat org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:61)\n\tat org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:94)\n\tat org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:623)\n\tat java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1128)\n\tat java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:628)\n\tat java.base/java.lang.Thread.run(Thread.java:829)\n\nDriver stacktrace:\n\tat org.apache.spark.scheduler.DAGScheduler.failJobAndIndependentStages(DAGScheduler.scala:2844)\n\tat org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$2(DAGScheduler.scala:2780)\n\tat org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$2$adapted(DAGScheduler.scala:2779)\n\tat scala.collection.mutable.ResizableArray.foreach(ResizableArray.scala:62)\n\tat scala.collection.mutable.ResizableArray.foreach$(ResizableArray.scala:55)\n\tat scala.collection.mutable.ArrayBuffer.foreach(ArrayBuffer.scala:49)\n\tat org.apache.spark.scheduler.DAGScheduler.abortStage(DAGScheduler.scala:2779)\n\tat org.apache.spark.scheduler.DAGScheduler.$anonfun$handleTaskSetFailed$1(DAGScheduler.scala:1242)\n\tat org.apache.spark.scheduler.DAGScheduler.$anonfun$handleTaskSetFailed$1$adapted(DAGScheduler.scala:1242)\n\tat scala.Option.foreach(Option.scala:407)\n\tat org.apache.spark.scheduler.DAGScheduler.handleTaskSetFailed(DAGScheduler.scala:1242)\n\tat org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.doOnReceive(DAGScheduler.scala:3048)\n\tat org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:2982)\n\tat org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:2971)\n\tat org.apache.spark.util.EventLoop$$anon$1.run(EventLoop.scala:49)\nCaused by: java.lang.RuntimeException: Labels MUST be Integers, but got 4.5\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.hashAgg_doAggregate_max_0$(Unknown Source)\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.hashAgg_doConsume_0$(Unknown Source)\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.hashAgg_doAggregateWithoutKey_0$(Unknown Source)\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.processNext(Unknown Source)\n\tat org.apache.spark.sql.execution.BufferedRowIterator.hasNext(BufferedRowIterator.java:43)\n\tat org.apache.spark.sql.execution.WholeStageCodegenEvaluatorFactory$WholeStageCodegenPartitionEvaluator$$anon$1.hasNext(WholeStageCodegenEvaluatorFactory.scala:43)\n\tat scala.collection.Iterator$$anon$10.hasNext(Iterator.scala:460)\n\tat org.apache.spark.shuffle.sort.BypassMergeSortShuffleWriter.write(BypassMergeSortShuffleWriter.java:140)\n\tat org.apache.spark.shuffle.ShuffleWriteProcessor.write(ShuffleWriteProcessor.scala:59)\n\tat org.apache.spark.scheduler.ShuffleMapTask.runTask(ShuffleMapTask.scala:104)\n\tat org.apache.spark.scheduler.ShuffleMapTask.runTask(ShuffleMapTask.scala:54)\n\tat org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:161)\n\tat org.apache.spark.scheduler.Task.run(Task.scala:141)\n\tat org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$4(Executor.scala:620)\n\tat org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:64)\n\tat org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:61)\n\tat org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:94)\n\tat org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:623)\n\tat java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1128)\n\tat java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:628)\n\tat java.base/java.lang.Thread.run(Thread.java:829)\n",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mPy4JJavaError\u001b[0m                             Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-150-39dc31e86b98>\u001b[0m in \u001b[0;36m<cell line: 47>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     45\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     46\u001b[0m \u001b[0;31m# # Train the model\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 47\u001b[0;31m \u001b[0mdt_model\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdt_model\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtraining_data\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     48\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     49\u001b[0m \u001b[0;31m# # Make predictions on the testing set\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pyspark/ml/base.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, dataset, params)\u001b[0m\n\u001b[1;32m    203\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcopy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mparams\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_fit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    204\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 205\u001b[0;31m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_fit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    206\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    207\u001b[0m             raise TypeError(\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pyspark/ml/wrapper.py\u001b[0m in \u001b[0;36m_fit\u001b[0;34m(self, dataset)\u001b[0m\n\u001b[1;32m    379\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    380\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_fit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdataset\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mDataFrame\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m->\u001b[0m \u001b[0mJM\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 381\u001b[0;31m         \u001b[0mjava_model\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_fit_java\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    382\u001b[0m         \u001b[0mmodel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_create_model\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mjava_model\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    383\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_copyValues\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pyspark/ml/wrapper.py\u001b[0m in \u001b[0;36m_fit_java\u001b[0;34m(self, dataset)\u001b[0m\n\u001b[1;32m    376\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    377\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_transfer_params_to_java\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 378\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_java_obj\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_jdf\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    379\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    380\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_fit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdataset\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mDataFrame\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m->\u001b[0m \u001b[0mJM\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/py4j/java_gateway.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *args)\u001b[0m\n\u001b[1;32m   1320\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1321\u001b[0m         \u001b[0manswer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgateway_client\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msend_command\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcommand\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1322\u001b[0;31m         return_value = get_return_value(\n\u001b[0m\u001b[1;32m   1323\u001b[0m             answer, self.gateway_client, self.target_id, self.name)\n\u001b[1;32m   1324\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pyspark/errors/exceptions/captured.py\u001b[0m in \u001b[0;36mdeco\u001b[0;34m(*a, **kw)\u001b[0m\n\u001b[1;32m    177\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mdeco\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0ma\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mAny\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkw\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mAny\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m->\u001b[0m \u001b[0mAny\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    178\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 179\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mf\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0ma\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkw\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    180\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mPy4JJavaError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    181\u001b[0m             \u001b[0mconverted\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mconvert_exception\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0me\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mjava_exception\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/py4j/protocol.py\u001b[0m in \u001b[0;36mget_return_value\u001b[0;34m(answer, gateway_client, target_id, name)\u001b[0m\n\u001b[1;32m    324\u001b[0m             \u001b[0mvalue\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mOUTPUT_CONVERTER\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mtype\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0manswer\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgateway_client\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    325\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0manswer\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0mREFERENCE_TYPE\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 326\u001b[0;31m                 raise Py4JJavaError(\n\u001b[0m\u001b[1;32m    327\u001b[0m                     \u001b[0;34m\"An error occurred while calling {0}{1}{2}.\\n\"\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    328\u001b[0m                     format(target_id, \".\", name), value)\n",
            "\u001b[0;31mPy4JJavaError\u001b[0m: An error occurred while calling o10841.fit.\n: org.apache.spark.SparkException: Job aborted due to stage failure: Task 0 in stage 671.0 failed 1 times, most recent failure: Lost task 0.0 in stage 671.0 (TID 693) (f139c0dcbf82 executor driver): java.lang.RuntimeException: Labels MUST be Integers, but got 4.5\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.hashAgg_doAggregate_max_0$(Unknown Source)\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.hashAgg_doConsume_0$(Unknown Source)\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.hashAgg_doAggregateWithoutKey_0$(Unknown Source)\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.processNext(Unknown Source)\n\tat org.apache.spark.sql.execution.BufferedRowIterator.hasNext(BufferedRowIterator.java:43)\n\tat org.apache.spark.sql.execution.WholeStageCodegenEvaluatorFactory$WholeStageCodegenPartitionEvaluator$$anon$1.hasNext(WholeStageCodegenEvaluatorFactory.scala:43)\n\tat scala.collection.Iterator$$anon$10.hasNext(Iterator.scala:460)\n\tat org.apache.spark.shuffle.sort.BypassMergeSortShuffleWriter.write(BypassMergeSortShuffleWriter.java:140)\n\tat org.apache.spark.shuffle.ShuffleWriteProcessor.write(ShuffleWriteProcessor.scala:59)\n\tat org.apache.spark.scheduler.ShuffleMapTask.runTask(ShuffleMapTask.scala:104)\n\tat org.apache.spark.scheduler.ShuffleMapTask.runTask(ShuffleMapTask.scala:54)\n\tat org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:161)\n\tat org.apache.spark.scheduler.Task.run(Task.scala:141)\n\tat org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$4(Executor.scala:620)\n\tat org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:64)\n\tat org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:61)\n\tat org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:94)\n\tat org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:623)\n\tat java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1128)\n\tat java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:628)\n\tat java.base/java.lang.Thread.run(Thread.java:829)\n\nDriver stacktrace:\n\tat org.apache.spark.scheduler.DAGScheduler.failJobAndIndependentStages(DAGScheduler.scala:2844)\n\tat org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$2(DAGScheduler.scala:2780)\n\tat org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$2$adapted(DAGScheduler.scala:2779)\n\tat scala.collection.mutable.ResizableArray.foreach(ResizableArray.scala:62)\n\tat scala.collection.mutable.ResizableArray.foreach$(ResizableArray.scala:55)\n\tat scala.collection.mutable.ArrayBuffer.foreach(ArrayBuffer.scala:49)\n\tat org.apache.spark.scheduler.DAGScheduler.abortStage(DAGScheduler.scala:2779)\n\tat org.apache.spark.scheduler.DAGScheduler.$anonfun$handleTaskSetFailed$1(DAGScheduler.scala:1242)\n\tat org.apache.spark.scheduler.DAGScheduler.$anonfun$handleTaskSetFailed$1$adapted(DAGScheduler.scala:1242)\n\tat scala.Option.foreach(Option.scala:407)\n\tat org.apache.spark.scheduler.DAGScheduler.handleTaskSetFailed(DAGScheduler.scala:1242)\n\tat org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.doOnReceive(DAGScheduler.scala:3048)\n\tat org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:2982)\n\tat org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:2971)\n\tat org.apache.spark.util.EventLoop$$anon$1.run(EventLoop.scala:49)\nCaused by: java.lang.RuntimeException: Labels MUST be Integers, but got 4.5\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.hashAgg_doAggregate_max_0$(Unknown Source)\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.hashAgg_doConsume_0$(Unknown Source)\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.hashAgg_doAggregateWithoutKey_0$(Unknown Source)\n\tat org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.processNext(Unknown Source)\n\tat org.apache.spark.sql.execution.BufferedRowIterator.hasNext(BufferedRowIterator.java:43)\n\tat org.apache.spark.sql.execution.WholeStageCodegenEvaluatorFactory$WholeStageCodegenPartitionEvaluator$$anon$1.hasNext(WholeStageCodegenEvaluatorFactory.scala:43)\n\tat scala.collection.Iterator$$anon$10.hasNext(Iterator.scala:460)\n\tat org.apache.spark.shuffle.sort.BypassMergeSortShuffleWriter.write(BypassMergeSortShuffleWriter.java:140)\n\tat org.apache.spark.shuffle.ShuffleWriteProcessor.write(ShuffleWriteProcessor.scala:59)\n\tat org.apache.spark.scheduler.ShuffleMapTask.runTask(ShuffleMapTask.scala:104)\n\tat org.apache.spark.scheduler.ShuffleMapTask.runTask(ShuffleMapTask.scala:54)\n\tat org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:161)\n\tat org.apache.spark.scheduler.Task.run(Task.scala:141)\n\tat org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$4(Executor.scala:620)\n\tat org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:64)\n\tat org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:61)\n\tat org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:94)\n\tat org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:623)\n\tat java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1128)\n\tat java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:628)\n\tat java.base/java.lang.Thread.run(Thread.java:829)\n"
          ]
        }
      ]
    }
  ]
}