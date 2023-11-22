#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright ©2023 Derwen, Inc. All rights reserved.

"""
HuggingFace Spaces using Streamlit
"""

import streamlit as st


if __name__ == "__main__":
    x = st.slider("Select a value")
    st.write(x, "squared is", x * x)
