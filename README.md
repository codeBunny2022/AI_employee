# AI_employee

For data preprocessing used AutoClean

Citation-https://github.com/elisemercury/AutoClean.git

```
pip install py-AutoClean
```

## Explaination video


## Demo Video

[Download and watch the demo video](assignment_explaination_chirag.mkv)
# AI Employee

## Project Overview

This project aims to analyze the provided dataset, perform detailed statistical analysis, and generate a comprehensive report. The report includes summaries, correlations, PCA plots, regression analysis, and value counts for categorical variables. The entire process is automated through a command-line interface (CLI) that allows users to interact with the system, perform analysis, and generate reports.

## Tech Stack

- **Python**: Main programming language for the project.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **SciKit-Learn**: For machine learning tasks such as PCA, clustering, and regression analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **FPDF**: For generating PDF reports.
- **Hugging Face Transformers**: For natural language processing (NLP) tasks, specifically summarization and question-answering.
- **AutoClean**: For cleaning the dataset.

## Project Structure

```plaintext
.
├── analysis_engine.py
├── cli_interface.py
├── preprocessor.py
├── report_generator.py
├── README.md
└── olympics2024.csv
```
#### 1. preprocessor.py
- Purpose: Loads and cleans the dataset.
- Description:
  - load_data(file_path): Loads data from CSV, JSON, and Excel formats.
  - The dataset is cleaned using the AutoClean library, which handles duplicates, missing values, and datetime extraction.
- Usage:
  - Place the dataset file (e.g., olympics2024.csv) in the project directory.
  - Run the script to load and clean the dataset.

#### 2. analysis_engine.py
- Purpose: Performs detailed analysis on the dataset.
- Description:
  - ProfoundAnalysisEngine: A class that initializes with the data file path and contains methods to:
    - Load and initialize data.
    - Perform analysis, including handling categorical and numerical columns.
    - Conduct clustering and PCA (Principal Component Analysis).
    - Visualize results and save them as images.
  - The analysis results are saved in the analysis_results directory.
- Usage:
  - Ensure the cleaned dataset is available.
  - Run the script to perform the analysis.

#### 3. report_generator.py
- Purpose: Generates a PDF report summarizing the analysis results.
- Description:
  - ReportGenerator: A class with methods to:
    - Add titles, sections, paragraphs, and images to the report.
    - Generate sections with summarized text and images.
    - Save the report to a specified path.
  - Utilizes FPDF for report creation and Hugging Face Transformers for text summarization.
- Usage:
  - Ensure the analysis results are available in the analysis_results directory.
  - Run the script to generate a PDF report. The report will be saved as olympics_analysis_report.pdf.

#### 4. cli_interface.py
- Purpose: Provides a command-line interface for interaction.
- Description:
  - AIEmployeeCLI: A class that implements:
    - Initialization and running of the command-line interface.
    - Handling user commands to analyze data, generate reports, provide summaries, and answer queries.
  - Uses NLP models for summarizing data and answering questions.
- Usage:
  - Run the script to start the CLI interface.
  - Use the provided commands to interact with the project.


## How to use

### 1. Pre-Requisites

Ensure you have Python installed along with the necessary libraries. Install the required libraries using the following command:

```
pip install pandas numpy scikit-learn matplotlib seaborn fpdf transformers autoclean
```

### 2. Use the CLI

```
python cli_interface.py
```

### CLI Commands

- help: Show available commands.
- analyze data: Perform analysis on the dataset.
- generate report: Generate a comprehensive report.
- summary: Show a summary of the analyzed data.
- You can also ask questions directly about the dataset analysis.
