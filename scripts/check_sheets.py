import pandas as pd
excel_path = "docs/2025年8月+花椒挥发油测定结果.xls"
try:
    xl = pd.ExcelFile(excel_path)
    print(xl.sheet_names)
except Exception as e:
    print(e)
