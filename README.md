# RNASeq-MachineLearning-Project
Comprehensive RNA-seq and machine learning analysis pipeline for identifying and visualizing differentially expressed genes and patterns in transcriptomic data using R and Python.

Workflow Summary

Part 1: Differential Expression Analysis (R)

Load Required Packages
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c("DESeq2", "airway"))
install.packages(c("ggplot2", "pheatmap"))

library(DESeq2)
library(ggplot2)
library(pheatmap)
library(airway)

Load and Inspect Data
data("airway")
head(airway)


Dataset: airway smooth muscle cells (treated vs untreated).

Create DESeqDataSet and Preprocess
dds <- DESeqDataSetFromMatrix(
    countData = assay(airway),
    colData = colData(airway),
    design = ~ cell + dex
)

dds <- dds[rowSums(counts(dds)) > 10, ]   # Filter low counts
dds <- DESeq(dds)                         # Normalize and model

Differential Expression Analysis
res <- results(dds)
resLFC <- lfcShrink(dds, coef="dex_trt_vs_untrt", type="apeglm")
summary(resLFC)

Visualization
Boxplot
boxplot(log2(counts(dds) + 1), main="Gene Expression Distribution")
<img width="431" height="246" alt="Rplot04" src="https://github.com/user-attachments/assets/df2aea82-5376-46ad-b734-d9203e514c44" />

MA Plot
plotMA(resLFC, main="MA Plot: Treated vs Untreated", ylim=c(-5,5))
<img width="431" height="246" alt="Rplot03" src="https://github.com/user-attachments/assets/81cc58fe-4116-4f62-9c8b-5b0ad707ca1b" />

Volcano Plot (optional)
ggplot(as.data.frame(res), aes(x=log2FoldChange, y=-log10(pvalue))) +
  geom_point(alpha=0.4) +
  theme_minimal() +
  ggtitle("Volcano Plot of Differential Expression")
<img width="431" height="246" alt="Rplot" src="https://github.com/user-attachments/assets/3ed17220-9014-40d7-98f8-a87581b68686" />


PCA Analysis
vsd <- vst(dds, blind=FALSE)
plotPCA(vsd, intgroup="dex")
<img width="431" height="246" alt="Rplot02" src="https://github.com/user-attachments/assets/6e37308f-fa57-4752-a560-dddd016114f8" />

Heatmap of Top Variable Genes
topVarGenes <- head(order(rowVars(assay(vsd)), decreasing=TRUE), 50)
pheatmap(assay(vsd)[topVarGenes,],
         cluster_rows=TRUE, show_rownames=FALSE,
         cluster_cols=TRUE, annotation_col=colData(vsd))
<img width="431" height="246" alt="Rplot01" src="https://github.com/user-attachments/assets/00c66a97-28f9-4187-8063-33605df52558" />

Save Results
write.csv(as.data.frame(resLFC), "results/differential_expression_results.csv")
ggsave("plots/volcano_plot.png", width=6, height=5)

Learning Outcomes
Hands-on experience with DESeq2 for RNA-seq differential expression
Understanding of normalization, fold change, and p-value adjustment
Practice in data visualization (MA, PCA, heatmap)
Insights into biological interpretation of gene expression results



🐍 Part 2 — Python Machine Learning Analysis

This section continues the RNA-seq pipeline by applying machine learning to the DESeq2 results exported from R.
The goal was to classify samples based on their gene expression patterns using a Random Forest model and assess the performance with Leave-One-Out Cross Validation (LOOCV).

Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

Load the Data
data = pd.read_csv('DESeq2_significant_results.csv', index_col=0)


Note: Make sure the file DESeq2_significant_results.csv (exported from R) is in your Python working directory.

data.shape
data.head()
data.info()

PCA Visualization of Significant Genes
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data.values)

plt.figure(figsize=(8,6))
plt.scatter(pca_result[:,0], pca_result[:,1], alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of RNA-seq Significant Genes')
plt.show()


Purpose: Visualizes how samples cluster based on gene expression variation.

Load and Prepare Normalized Counts
norm_counts = pd.read_csv('normalized_counts.csv', index_col=0)
norm_counts.shape
norm_counts.head()

# Transpose so rows = samples, columns = genes
norm_counts = norm_counts.T

Scale the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(norm_counts.values)
print(X_scaled.shape)

Purpose: Standardizes the features so each gene contributes equally to the analysis.

Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, labels, test_size=0.25, random_state=42, stratify=labels
)


Note: labels should contain the biological condition for each sample (e.g., treated vs untreated).

Random Forest Model with LOOCV
model = RandomForestClassifier(random_state=42)
loo = LeaveOneOut()

scores = cross_val_score(model, X_scaled, labels, cv=loo)
print("LOOCV Accuracy per sample:", scores)
print("Mean LOOCV Accuracy:", scores.mean())


Outcome:
LOOCV Accuracy per sample: [1. 1. 1. 1. 1. 1. 1. 1.]
Mean LOOCV Accuracy: 1.0


Interpretation: Perfect accuracy likely reflects overfitting due to the very small sample size (8 samples). The workflow demonstrates the concept, not a deployable model.

Learning Outcomes
Learned how to import DESeq2 results and integrate R → Python workflows
Applied data preprocessing and scaling for ML
Used PCA for dimensionality reduction and visualization
Built a Random Forest model and validated with Leave-One-Out Cross Validation
Understood overfitting issues in small biological datasets

Results Summary

The analysis successfully identified and visualized gene expression differences between dexamethasone-treated and control airway smooth muscle cells using DESeq2 and supporting Python tools.

Key Findings:
Differential Expression: Several genes were found to be significantly upregulated or downregulated after treatment, based on adjusted p-values and log2 fold change.
Data Distribution: Boxplots confirmed consistent normalization across samples.
MA Plot: Showed clear separation between genes with significant expression changes and those with minimal differences.
PCA Plot: Samples clustered according to treatment groups, confirming distinct transcriptional profiles.
Heatmap: Highlighted clusters of co-expressed genes, revealing potential gene groups affected by the treatment.
Machine Learning Validation: Random Forest classification achieved perfect prediction accuracy (Mean LOOCV Accuracy = 1.0), supporting the distinct expression patterns identified in the R analysis.

Output Files:
DESeq2_significant_results.csv – Differential expression results
normalized_counts.csv – Normalized read counts used for downstream analysis
Plots – PCA, heatmap, and MA plot images for visual interpretation
