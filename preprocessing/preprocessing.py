"""
converting the reports.csv file into a dictionary to use it in build_database.py

csv --> langchain_core.documents.base.Document

Should look something like this, so I need to extract the page content and its metadata:
document = Document(
    page_content="Hello, world!",
    metadata={"source": "https://example.com"}
)

"""

import csv
from langchain_core.documents import Document
import re
import pandas as pd

filename = "reports.csv"

# TODO: In the future, I think we can add a function for additional metadata related to the keywords in all caps
# I will likely resubmit this later with additional preprocessing because of that, hopefully within the next few days

# report = "PREOPERATIVE DIAGNOSIS: , Secondary capsular membrane, right eye.,POSTOPERATIVE DIAGNOSIS: , Secondary capsular membrane, right eye.,PROCEDURE PERFORMED: , YAG laser capsulotomy, right eye.,INDICATIONS: , This patient has undergone cataract surgery, and vision is reduced in the operated eye due to presence of a secondary capsular membrane.  The patient is being brought in for YAG capsular discission.,PROCEDURE: , The patient was seated at the YAG laser, the pupil having been dilated with 1%  Mydriacyl, and Iopidine was instilled.  The Abraham capsulotomy lens was then positioned and applications of laser energy in the pattern indicated on the outpatient note were applied.  A total of"
report = "2-D M-MODE: , ,1.  Left atrial enlargement with left atrial diameter of 4.7 cm.,2.  Normal size right and left ventricle.,3.  Normal LV systolic function with left ventricular ejection fraction of 51%.,4.  Normal LV diastolic function.,5.  No pericardial effusion.,6.  Normal morphology of aortic valve, mitral valve, tricuspid valve, and pulmonary valve.,7.  PA systolic pressure is 36 mmHg.,DOPPLER: , ,1.  Mild mitral and tricuspid regurgitation.,2.  Trace aortic and pulmonary regurgitation."
# report = "PREOPERATIVE DIAGNOSIS: , Secondary capsular membrane, right eye.,POSTOPERATIVE DIAGNOSIS: , Secondary capsular membrane, right eye.,PROCEDURE WAS PERFORMED: , YAG laser capsulotomy, right eye.,INDICATIONS: , This patient has undergone cataract surgery, and vision is reduced in the operated eye due to presence of a secondary capsular membrane.  The patient is being brought in for YAG capsular discission.,PROCEDURE: , The patient was seated at the YAG laser, the pupil having been dilated with 1%  Mydriacyl, and Iopidine was instilled.  The Abraham capsulotomy lens was then positioned and applications of laser energy in the pattern indicated on the outpatient note were applied.  A total of"

def split_report(report):
    """
    In addition to the csv category on the medical specialty, more general information is categorized in the reports
    It always is "[ALL CAPS CATEGORY]:", so we will split the report into these categories.
    """
    initial = re.split(r'(?=\b[A-Z]+:)', report)

    keys = []
    values = []
    for i in range(len(initial)):
        # create a dictionary from this
        if ":" in initial[i]:
            words_initial = initial[i].split(" ")
            category = words_initial[0]  # get the last word of the category name
            description = initial[i].removeprefix(category) # get the description for the category
            if i > 0:  # getting any other capital letters from before the last one if they existed
                match = re.search(r'([\dA-Z-]+(?:\s+[\dA-Z-]+)*)$', initial[i - 1])  # pattern two
                if match is None:
                    match = re.search(r'([A-Z]+(?:\s+[A-Z]+)*)\W*$', initial[i - 1])  # pattern one
                if match:
                    category = match.group(1) + " " + category  # getting the full category name
                    # print(f"category: {category}")
                    if len(values) > 0:
                        values[-1] = values[-1].removesuffix(match.group(1)+" ")  # removing the first part of the category from the last description
                        # print(f"match: {match.group(1)}")
                        # print(f"values: {values}")
                    keys.append(category)
                    values.append(description)

    metadata = dict(zip(keys, values))

    return metadata

def create_documents():
    new_docs = []

    with open(filename) as csvfile:
        df = pd.read_csv(csvfile)
        for index, row in df.iterrows():
            additional_metadata = split_report(row["report"])
            additional_metadata["medical_specialty"] = row["medical_specialty"]
            document = Document(
                page_content=row["report"],
                metadata=additional_metadata
            )
            new_docs.append(document)

    return new_docs

# split_report = split_report(report)
# print(split_report)
# for report in split_report:
#     print()
#     print(report)
documents = create_documents()
print(documents[9])

# i think the general step should be going through the csv
# for each row, get the page content, also get all of the metadata. For now, I think we could get the columns and if we're feeling crazy we can get the capitalized words too as metadata
# I think then we just create a list of documents, so we can add them all to the database?
# after that to we have to convert them to FAISS or something? nope, then we're done
# something like this: document_ids = vector_store.add_documents(documents=all_splits)

