library(tidyr)
library(dplyr)
library(rstatix)

# Load data
df <- read.csv("results.csv")

# Check for normality using the Shapiro-Wilk test for each metric
shapiro_test_results <- df %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F1_Score), 
               names_to = "Metric", 
               values_to = "Value") %>%
  group_by(Model, Metric) %>%
  summarise(
    shapiro_p_value = shapiro.test(Value)$p.value,
    .groups = 'drop'
  )

# Save the Shapiro-Wilk test results
write.csv(shapiro_test_results, "shapiro_wilk_test_results.csv", row.names = FALSE)

# Check if all metrics passed normality
non_normal_metrics <- shapiro_test_results %>%
  filter(shapiro_p_value < 0.05)

# Save the metrics that failed normality
write.csv(non_normal_metrics, "non_normal_metrics.csv", row.names = FALSE)

if (nrow(non_normal_metrics) == 0) {
  # All metrics passed the normality test
  print("All metrics passed the normality test. Skipping Kruskal-Wallis test.")
  print("Consider performing an ANOVA if appropriate.")
} else {
  # Perform Kruskal-Wallis test for all metrics
  print("Some metrics failed the normality test. Performing Kruskal-Wallis test for all metrics.")
  
  # Prepare data for Kruskal-Wallis
  df_long <- df %>%
    pivot_longer(cols = c(Accuracy, Precision, Recall, F1_Score), 
                 names_to = "Metric", 
                 values_to = "Value")
  
  kruskal_results <- df_long %>%
    group_by(Metric) %>%
    kruskal_test(Value ~ Model) %>%
    adjust_pvalue(method = "holm") %>%
    add_significance()
  
  print("Kruskal-Wallis test results:")
  print(kruskal_results)
  
  # Save the Kruskal-Wallis results
  write.csv(kruskal_results, "kruskal_wallis_results.csv", row.names = FALSE)
  
  # Identify metrics with significant Kruskal-Wallis results
  significant_metrics <- kruskal_results %>%
    filter(p.adj < 0.05) %>%
    pull(Metric)
  
  if (length(significant_metrics) > 0) {
    # Perform Dunn's test for pairwise comparisons with Holm–Bonferroni correction
    print("At least one metric was significant in Kruskal-Wallis. Performing Dunn's test.")
    dunn_results <- df_long %>%
      filter(Metric %in% significant_metrics) %>%
      group_by(Metric) %>%
      dunn_test(Value ~ Model, p.adjust.method = "holm") %>%
      add_significance()
    
    print("Dunn's test results:")
    print(dunn_results)
    
    # Save the Dunn's test results
    write.csv(dunn_results, "dunn_test_results.csv", row.names = FALSE)
  } else {
    print("No metrics were significant in Kruskal-Wallis. Skipping Dunn's test.")
  }
}
