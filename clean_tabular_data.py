# Alan (Wentao Li), Imperial College London
# AICore 2022, All rights reserved

import pandas as pd

rawtext_images = pd.read_csv('raw_data/Images.csv', lineterminator="\n")
rawtext_products = pd.read_csv('raw_data/Products.csv', lineterminator="\n")

print(rawtext_images, rawtext_products)