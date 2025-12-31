import pandas as pd

excel_path = "docs/2025年8月+花椒挥发油测定结果.xls"
sheets = ["云南竹叶椒", "云南藤椒1"]

for sheet in sheets:
    print(f"--- Sheet: {sheet} ---")
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        print(f"Total rows: {len(df)}")
        print("Columns:", df.columns.tolist())
        print("First 5 rows:")
        print(df.head())
        
        # Check for potential issues with column names (e.g. whitespace)
        print("\nColumn name check:")
        for col in df.columns:
            print(f"'{col}'")
            
        # Check required columns
        distill_cols = ("蒸馏量（初）ml", "蒸馏量初ml", "蒸馏量", "蒸馏量_ml")
        weight_cols = ("重量", "重量(g)", "重量（g）", "重量g", "样品重量")
        id_cols = ("高光谱图件编号", "图件编号", "编号", "样本编号")
        
        def find_col(candidates):
            for c in candidates:
                if c in df.columns: return c
            return None
            
        d_col = find_col(distill_cols)
        w_col = find_col(weight_cols)
        i_col = find_col(id_cols)
        
        print(f"\nFound columns: ID='{i_col}', Weight='{w_col}', Distill='{d_col}'")
        
        if d_col and w_col:
            valid_rows = df.dropna(subset=[d_col, w_col])
            print(f"Rows with valid weight/distill: {len(valid_rows)}")
        else:
            print("Missing required columns!")
            
    except Exception as e:
        print(f"Error reading sheet: {e}")
    print("\n")
