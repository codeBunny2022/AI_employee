import os
import sys
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from analysis_engine import ProfoundAnalysisEngine
from report_generator import ReportGenerator

class AIEmployeeCLI:
    def __init__(self, data_file="olympics2024.csv"):
        self.data_file = data_file
        self.analysis_engine = ProfoundAnalysisEngine(data_file)
        self.summarizer = pipeline("summarization")
        self.question_answerer = pipeline("question-answering")
        print("AI Employee initialized. Type 'help' for a list of commands.")

    def run(self):
        while True:
            user_input = input("You: ").strip().lower()
            if user_input in ["exit", "quit"]:
                print("Goodbye!")
                break
            elif user_input == "help":
                self.show_help()
            elif "generate report" in user_input:
                self.generate_report()
            elif "analyze data" in user_input:
                self.analyze_data()
            elif "summary" in user_input:
                self.display_summary()
            else:
                self.answer_query(user_input)

    def show_help(self):
        help_texts = [
            "Commands:",
            "  help: Show this help message",
            "  analyze data: Perform analysis on the dataset",
            "  generate report: Generate a comprehensive report",
            "  summary: Show a summary of the analyzed data",
            "  exit/quit: Exit the program",
            "You can also ask questions about the data analysis."
        ]
        print("\n".join(help_texts))

    def analyze_data(self):
        self.analysis_engine.analyze()
        print("Data analysis completed.")

    def generate_report(self):
        report = ReportGenerator(analysis_dir="analysis_results", output_path="olympics_analysis_report.pdf")
        report.add_title("Olympics 2024 Data Analysis Report")
        
        analysis_details = {
            "Data Summary": {
                "text": "This section provides a summary of the dataset, with important statistical descriptions of both numerical and categorical features.",
                "images": ["data_summary.png"]  # Placeholder for actual summary visualization
            },
            "Correlation Heatmap": {
                "text": "The heatmap below illustrates the correlation coefficients between numerical features. Correlation values close to 1 signify strong positive relationships, while values closer to -1 signify strong negative relationships.",
                "images": ["correlation_heatmap.png"]
            },
            "PCA Plot": {
                "text": "Principal Component Analysis (PCA) is performed to reduce the dimensionality of the data. The plot below highlights the components with the highest variance.",
                "images": ["pca_plot.png"]
            },
            "Regression Analysis": {
                "text": "Linear regression analysis explores the relationship between the dependent variable and one or more independent variables. The plot demonstrates the fit line.",
                "images": [
                    "regression_plot_Bronze.png",
                    "regression_plot_Country Code_encoded.png",
                    "regression_plot_Country_encoded.png",
                    "regression_plot_Gold.png",
                    "regression_plot_PCA1.png",
                    "regression_plot_PCA2.png",
                    "regression_plot_Rank.png",
                    "regression_plot_Silver.png",
                    "regression_plot_Total.png"
                ]
            },
            "Categorical Value Counts": {
                "text": "The following plots show the count of various categories in the categorical columns of the dataset.",
                "images": [
                    "count_plot_Country Code.png",
                    "count_plot_Country.png"
                ]
            }
        }

        for section, details in analysis_details.items():
            report.generate_report_section(title=section, text=details["text"], image_filenames=details["images"])
        
        report.save_report()
        print("Report generated successfully.")

    def display_summary(self):
        summary_text = (
            "The dataset contains various details about the Olympics 2024 data, including measures of central tendency, "
            "correlation heatmaps, Principal Component Analysis (PCA) plots, linear regression analysis, and count "
            "plots for categorical variables."
        )
        summarized = self.summarizer(summary_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        print(f"Summary:\n{summarized}")

    def answer_query(self, query):
        context = (
            "The dataset analysis includes detailed statistical summaries, correlation matrices, PCA plots, linear regression models, "
            "and categorical value counts. Each section of the report provides insights into different aspects of the data."
        )
        result = self.question_answerer(question=query, context=context)
        print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    cli = AIEmployeeCLI()
    cli.run()