import pandas as pd

excel_path = "docs/2025年8月+花椒挥发油测定结果.xls"
try:
    # Read the summary sheet
    df_all = pd.read_excel(excel_path, sheet_name="挥发油")
    all_ids = set(df_all["高光谱图件编号"].dropna().astype(str).str.strip())
    print(f"'挥发油' sheet has {len(all_ids)} unique IDs.")
    
    # Read individual sheets
    sub_sheets = [
        '枸椒', '云南竹叶椒', '云南藤椒1', '贵州藤椒', '金佛山野生椒-落叶', 
        '江苏红花椒', '湖北金宏椒', '早熟九叶青', '韩城大红袍', '九叶青', 
        '奉节竹叶椒', '德阳青花椒', '大巴山野山椒', '金佛山CK3'
    ]
    
    sub_ids = set()
    for sheet in sub_sheets:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        ids = df["高光谱图件编号"].dropna().astype(str).str.strip()
        sub_ids.update(ids)
        
    print(f"Sub-sheets combined have {len(sub_ids)} unique IDs.")
    
    diff = sub_ids - all_ids
    if not diff:
        print("SUCCESS: '挥发油' sheet contains all IDs from sub-sheets.")
    else:
        print(f"WARNING: '挥发油' sheet is missing {len(diff)} IDs: {diff}")
        
except Exception as e:
    print(e)
