using System.Diagnostics;
using Accord.Math;
using Accord.Statistics;
using Microsoft.Azure.CognitiveServices.Vision.ComputerVision;
using Microsoft.Azure.CognitiveServices.Vision.ComputerVision.Models;

namespace AzureVectorImageSearch;

internal class Program
{
    // Add your Computer Vision subscription key and endpoint here
    private static readonly string subscriptionKey = "YOUR_COMPUTER_VISION_SUBSCRIPTION_KEY";
    private static readonly string endpoint = "YOUR_COMPUTER_VISION_ENDPOINT";

    private static async Task Main(string[] args)
    {
        var client = new ComputerVisionClient(new ApiKeyServiceClientCredentials(subscriptionKey))
        {
            Endpoint = endpoint
        };

        // Sample public domain image URLs from Flickr with associated labels
        var images = new Dictionary<string, string>
        {
            { "https://live.staticflickr.com/5800/30084705221_6001bbf1ba_b.jpg", "City" },
            { "https://live.staticflickr.com/4572/38004374914_6b686d708e_b.jpg", "Forest" },
            { "https://live.staticflickr.com/3446/3858223360_e57140dd23_b.jpg", "Ocean" },
            { "https://live.staticflickr.com/7539/16129656628_ddd1db38c2_b.jpg", "Mountains" },
            { "https://live.staticflickr.com/3168/2839056817_2263932013_b.jpg", "Desert" }
        };

        // Store the vectors and all unique tags found
        var imageVectors = new Dictionary<string, double[]>();
        var allTags = new HashSet<string>();

        // Extract features for all images and collect all unique tags
        var imageTagConfidenceMap = new Dictionary<string, Dictionary<string, double>>();
        foreach (var imageUrl in images.Keys)
        {
            var imageTags = await ExtractImageTags(client, imageUrl);
            imageTagConfidenceMap[imageUrl] = imageTags;
            allTags.UnionWith(imageTags.Keys); // Collect unique tags
            Console.WriteLine($"Extracted tags for {imageUrl}");
        }

        // Create fixed-size vectors for each image based on all unique tags
        foreach (var imageUrl in images.Keys)
        {
            var vector = CreateFixedSizeVector(allTags, imageTagConfidenceMap[imageUrl]);
            imageVectors[imageUrl] = vector;
        }

        // Perform a search for the most similar image to a query image
        var queryImage = "https://live.staticflickr.com/3697/8753467625_e19f53756c_b.jpg"; // New forest image
        var queryTags = await ExtractImageTags(client, queryImage); // Extract tags for the new image
        var queryVector = CreateFixedSizeVector(allTags, queryTags); // Create vector based on those tags

        FindMostSimilarImagePerMetric(queryVector, imageVectors,
            images); // Search for the closest match using all metrics
        // Perform benchmarking
        Benchmark(queryVector, imageVectors);
    }

    // Extracts tags and their confidence scores from an image
    private static async Task<Dictionary<string, double>> ExtractImageTags(ComputerVisionClient client, string imageUrl)
    {
        var features = new List<VisualFeatureTypes?> { VisualFeatureTypes.Tags };
        var analysis = await client.AnalyzeImageAsync(imageUrl, features);

        var tagsConfidence = new Dictionary<string, double>();
        foreach (var tag in analysis.Tags) tagsConfidence[tag.Name] = tag.Confidence;
        return tagsConfidence;
    }

    // Create a fixed-size vector based on all possible tags
    private static double[] CreateFixedSizeVector(HashSet<string> allTags, Dictionary<string, double> imageTags)
    {
        var vector = new double[allTags.Count];
        var index = 0;
        foreach (var tag in allTags) vector[index++] = imageTags.TryGetValue(tag, out var imageTag) ? imageTag : 0.0;
        return vector;
    }

    // Manhattan (L1 norm)
    private static double ManhattanDistance(double[] a, double[] b)
    {
        double sum = 0;
        for (var i = 0; i < a.Length; i++) sum += Math.Abs(a[i] - b[i]);
        return sum;
    }

    // Chebyshev (L∞ norm)
    private static double ChebyshevDistance(double[] a, double[] b)
    {
        double max = 0;
        for (var i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    // Mahalanobis distance using Accord.NET
    private static double MahalanobisDistance(double[] a, double[] b, double[][] vectors)
    {
        // Compute the covariance matrix using Accord.NET
        var covarianceMatrix = ComputeCovarianceMatrix(vectors);

        // Compute the inverse of the covariance matrix
        var inverseCovarianceMatrix = covarianceMatrix.Inverse();

        // Calculate the difference vector
        var diff = a.Subtract(b);

        // Calculate Mahalanobis distance
        var mahalanobisDistance = diff.Dot(inverseCovarianceMatrix.Dot(diff));

        return Math.Sqrt(mahalanobisDistance);
    }

    // Compute covariance matrix using Accord.NET
    private static double[,] ComputeCovarianceMatrix(double[][] data)
    {
        // Get the number of rows and columns
        var rows = data.Length;
        var cols = data[0].Length;

        // Create a rectangular (2D) array
        var rectangularData = new double[rows, cols];

        // Copy the data from the jagged array into the rectangular array
        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++)
            rectangularData[i, j] = data[i][j];

        // Compute the covariance matrix using Accord.NET
        return rectangularData.Covariance();
    }

    // Find the most similar image for each distance metric
    private static void FindMostSimilarImagePerMetric(double[] queryVector, Dictionary<string, double[]> imageVectors,
        Dictionary<string, string> imageLabels)
    {
        var metrics = new Dictionary<string, Func<double[], double[], double>>
        {
            { "Euclidean", Distance.Euclidean },
            { "Manhattan", ManhattanDistance },
            { "Chebyshev", ChebyshevDistance },
            { "Mahalanobis", (a, b) => MahalanobisDistance(a, b, imageVectors.Values.ToArray()) }
        };

        // Find the most similar image for each metric
        foreach (var metric in metrics)
        {
            string mostSimilarImage = null;
            var smallestDistance = double.MaxValue;

            foreach (var imageVector in imageVectors)
            {
                var distance = metric.Value(queryVector, imageVector.Value);
                if (distance < smallestDistance)
                {
                    smallestDistance = distance;
                    mostSimilarImage = imageVector.Key; // Keep track of the image URL or ID
                }
            }

            // Check if the most similar image is found
            if (mostSimilarImage != null && imageLabels.ContainsKey(mostSimilarImage))
                Console.WriteLine(
                    $"Using {metric.Key} distance, this picture is most similar to: {imageLabels[mostSimilarImage]}");
            else
                Console.WriteLine($"No match found using {metric.Key} distance.");
        }
    }

    private static void Benchmark(double[] queryVector, Dictionary<string, double[]> imageVectors)
    {
        var metrics = new Dictionary<string, Func<double[], double[], double>>
        {
            { "Euclidean", Distance.Euclidean },
            { "Manhattan", ManhattanDistance },
            { "Chebyshev", ChebyshevDistance },
            { "Mahalanobis", (a, b) => MahalanobisDistance(a, b, imageVectors.Values.ToArray()) }
        };

        // Benchmark each metric
        foreach (var metric in metrics)
        {
            var stopwatch = new Stopwatch();
            stopwatch.Start();

            foreach (var imageVector in imageVectors) metric.Value(queryVector, imageVector.Value);

            stopwatch.Stop();
            Console.WriteLine($"{metric.Key} distance computation time: {stopwatch.ElapsedMilliseconds} ms");
        }
    }
}