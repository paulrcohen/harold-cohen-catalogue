#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 11:19:29 2025

@author: prcohen


# After editing files in Spyder and testing locally:

# 1. See what changed
git status

# 2. Add your changes
git add .

# 3. Save a snapshot with a message
git commit -m "Improved search algorithm and added new features"

# 4. Send to GitHub
git push

"""

import pandas as pd

df = pd.read_csv('./works/inventory_metadata.csv')

df.columns
