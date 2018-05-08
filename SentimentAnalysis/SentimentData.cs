using Microsoft.ML.Runtime.Api;

namespace SentimentAnalysis
{
    class SentimentData
    {
        [Column(ordinal: "0")]
        public string SentimentText;

        [Column(ordinal: "1", name: "Label")]
        public float Sentiment;
    }

    class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Sentiment;
    }
}
