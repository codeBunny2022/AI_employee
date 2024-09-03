import os
from fpdf import FPDF
from transformers import pipeline

class ReportGenerator:
    def __init__(self, analysis_dir="analysis_results", output_path="report.pdf"):
        """
        Initialize the report generator with output settings.
        
        Args:
            analysis_dir (str): Path to the directory containing analysis results.
            output_path (str): Path where the report will be saved.
        """
        self.pdf = FPDF()
        self.analysis_dir = analysis_dir
        self.output_path = output_path
        self.summarizer = pipeline("summarization")

    def add_title(self, title):
        """
        Add a title to the report.
        
        Args:
            title (str): The title of the report.
        """
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=24)
        self.pdf.cell(200, 10, txt=title, ln=True, align="C")
        self.pdf.ln(20)

    def add_section_title(self, section_title):
        """
        Add a section title to the report.
        
        Args:
            section_title (str): The title of the section.
        """
        self.pdf.set_font("Arial", size=18)
        self.pdf.cell(200, 10, txt=section_title, ln=True, align="L")
        self.pdf.ln(10)

    def add_paragraph(self, text):
        """
        Add a paragraph of text to the report.
        
        Args:
            text (str): The paragraph text.
        """
        self.pdf.set_font("Arial", size=12)
        self.pdf.multi_cell(0, 10, txt=text)
        self.pdf.ln(10)

    def add_image(self, image_path, x=None, y=None, w=0, h=0):
        """
        Add an image to the report.
        
        Args:
            image_path (str): The path to the image file.
            x (float): x position.
            y (float): y position.
            w (float): Width of the image.
            h (float): Height of the image.
        """
        if os.path.exists(image_path):
            self.pdf.image(image_path, x=x, y=y, w=w, h=h)
            self.pdf.ln(10)
        else:
            print(f"Warning: Image file {image_path} not found")

    def generate_report_section(self, title, text, image_filenames):
        """
        Generate a section of the report with images and summarized text.
        
        Args:
            title (str): The section's title.
            text (str): The section's text for summarization.
            image_filenames (list): List of image filenames to be added with the section.
        """
        if not image_filenames:
            print(f"Warning: No images for section {title}, skipping section.")
            return

        summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        self.add_section_title(title)
        self.add_paragraph(summary)
        for image_filename in image_filenames:
            image_path = os.path.join(self.analysis_dir, image_filename)
            self.add_image(image_path, w=180)

    def save_report(self):
        """
        Save the report to the specified output path.
        """
        self.pdf.output(self.output_path)


def main():
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
    print("Report saved successfully!")


if __name__ == "__main__":
    main()