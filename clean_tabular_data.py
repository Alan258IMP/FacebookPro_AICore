# Alan (Wentao Li), Imperial College London
# AICore 2022, All rights reserved

import pandas as pd

def clean(table: pd.DataFrame):
    '''
    Clean the data in products.csv.
    Specifically, by converting prices into np.float64
    '''
    table['price'] = table['price'].str.replace('£','')
    table['price'] = table['price'].str.replace(',','')
    table['price'] = table['price'].astype('float64')
    return table

if __name__ == '__main__':
    product_table = pd.read_csv('raw_data/Products.csv', lineterminator = "\n")
    product_table_cleaned = clean(product_table)
    product_table_cleaned.to_csv('raw_data/Products_cleaned.csv', columns = ['id', 'product_name', 'category', 'product_description', 'price', 'location'])