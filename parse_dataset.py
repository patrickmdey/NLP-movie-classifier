import pandas as pd

with open("movies.txt", "r", encoding="utf-8", errors="ignore") as m_file:
    with open("movies.csv", "w") as s_file:
        s_file.write("review_summary;review_text;score\n")
        review_summary = ""
        review_text = ""
        score = ""
        for line in m_file:
            try:
                if "<br />" in line:
                    line = line.replace("<br />", "")
                if ";" in line:
                    line = line.replace(";", "")
                if "review/score" in line:
                    score = line.split(":")[1].strip()
                if "review/summary" in line:
                    review_summary = line.split(":")[1].strip()
                if "review/text" in line:
                    review_text = line.split(":")[1].strip()
                    s_file.write(review_summary + ";" + review_text + ";" + score + "\n")
                    review_summary = ""
                    review_text = ""
                    score = ""
            except UnicodeDecodeError:
                continue