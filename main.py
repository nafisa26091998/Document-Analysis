import streamlit as st

st.set_page_config(
    page_title="Main Page",
)
st.title('DocANALYSIS!')
st.caption('Powered by GPT3')
# st.sidebar.success("Select a page above.")
st.write('Welcome! This is your first step to reduce your efforts in reading a complete research paper.')
st.header('INTRODUCTION')
st.write('The analysis of medical research papers can be a time-consuming and labor-intensive process, particularly when trying to identify relevant papers and extract key information from them. However, by leveraging advanced technologies such as natural language processing and machine learning, it is now possible to reduce time consumption and manual efforts while reading medical research papers. DocAnalysis platform could  proposes a complete paper analysis, drug comparisons, and question and answering mechanisms to the user. This platform  utilize natural language processing algorithms (GPT3) to extract key information from medical research papers and present it in a clear and concise format. By using such a platform, users can save time and effort in searching for relevant papers and digging deeper into the paper, allowing them to focus on the analysis and interpretation of the data presented. Overall, this approach has the potential to revolutionize the way medical research papers are analyzed, providing researchers with a powerful tool for staying up-to-date with the latest research in their respective therapy areas.')
st.subheader('What are some of the pain points researchers face today?')
st.write('''Researchers may face several difficulties when reading a medical research paper. Some of these difficulties include:

1. Technical language: Medical research papers often contain technical language that can be difficult for researchers who are not familiar with the subject matter.

2. Complex statistical analysis: Many medical research papers include complex statistical analyses that can be difficult for researchers to understand and interpret.

3. Bias: Medical research papers may be biased, either due to funding sources, conflicts of interest, or other factors. Researchers must be able to recognize and account for bias when evaluating the results of a study.

4. Reproducibility: Researchers may have difficulty reproducing the results of a medical research study due to differences in study design, patient populations, or other factors.

5. Limited access to full-text articles: Researchers may have difficulty accessing the full text of medical research papers, particularly if they are not affiliated with a university or research institution.

6. Time constraints: Researchers may have limited time to read and analyze medical research papers, particularly if they are working on multiple projects simultaneously.

7. Lack of critical appraisal skills: Researchers may lack the critical appraisal skills necessary to assess the quality and validity of medical research papers.

Overall, reading medical research papers can be challenging, but with careful attention to detail and the development of critical appraisal skills, researchers can overcome these difficulties and effectively evaluate the quality and relevance of the research presented.''')

st.subheader('What are some of the analysis done on a research paper?')
st.write('''There are several types of analyses that can be performed on a biomedical research paper. Some of these include:

1. Study design analysis: This involves examining the study design used in the research paper to assess the validity of the conclusions drawn from the study.

2. Statistical analysis: This involves examining the statistical methods used in the research paper to ensure that the results are valid and reliable.

3. Data analysis: This involves examining the data presented in the research paper to assess the validity and reliability of the results.

4. Literature review analysis: This involves examining the literature cited in the research paper to assess the relevance and quality of the sources used.

5. Interpretation analysis: This involves examining the conclusions drawn from the research paper to assess their validity and relevance.

6. Ethical analysis: This involves examining the ethical considerations involved in the research paper to ensure that the study was conducted in a responsible and ethical manner.

7. Clinical relevance analysis: This involves examining the clinical relevance of the research findings and their potential impact on patient care.

Overall, these analyses are important for assessing the quality and validity of the research presented in a biomedical research paper.''')



page_bg_img = '''
<style>
.stApp {
background-image: url("https://images.pond5.com/white-pills-oxycodone-or-tylenols-012229355_prevstill.jpeg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)         
        
        

    
