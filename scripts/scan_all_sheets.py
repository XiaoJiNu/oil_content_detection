import pandas as pd

excel_path = "docs/2025年8月+花椒挥发油测定结果.xls"
try:
    xl = pd.ExcelFile(excel_path)
    sheet_names = xl.sheet_names
    
    valid_sheets = []
    
    print(f"Scanning {len(sheet_names)} sheets in {excel_path}...\n")
    
    for sheet in sheet_names:
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet)
            cols = df.columns.tolist()
            
            # Check for key columns roughly
            has_id = any("图件编号" in str(c) or "编号" in str(c) for c in cols)
            has_weight = any("重量" in str(c) for c in cols)
            has_distill = any("蒸馏量" in str(c) for c in cols)
            
            # Specifically check for the column used in previous logic
            exact_id_col = "高光谱图件编号" in cols
            
            if has_id and has_weight and has_distill:
                count = len(df)
                status = "POTENTIAL"
                if exact_id_col:
                    status = "MATCH"
                    valid_sheets.append((sheet, count))
                
                print(f"[{status}] Sheet '{sheet}': {count} rows. Cols: {cols}")
            else:
                print(f"[SKIP] Sheet '{sheet}': Missing columns. Cols: {cols}")
                
        except Exception as e:
            print(f"[ERROR] Sheet '{sheet}': {e}")

    print(f"\nSummary: Found {len(valid_sheets)} sheets with exact '高光谱图件编号' column:")
    for name, count in valid_sheets:
        print(f"  - {name}: {count} rows")
        
except Exception as e:
    print(e)
