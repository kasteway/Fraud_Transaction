# Import streamlit
import streamlit as st


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import precision_recall_curve, recall_score
from sklearn import metrics



# Custom Color/Style
html_page = """
 <div style="background-color:Thistle;padding:20px">
  <FONT color="#FFFFFF" size="55">Detect Fraud Transactions</FONT>
   
    
 </div>

 """

st.markdown(html_page,unsafe_allow_html=True)


# Images
from PIL import Image
img = Image.open("fraud2.png")
st.image(img,width=700,caption="Fraud Data Image from https://www.signifyd.com/blog/ecommerce-fraud-more-costly-than-you-know/")



# Intro
st.write(
    """ This RandomForest Classification model for fraud transactions can bring signficant savings 
    for a company by giving them the power to take action to save fraudulent transactions. 

    **Please visit the GitHub link at the bottom of this page for more information  
"""


)





with open("models/X_pro_results.npy", "rb") as f:
    x_prob_results = np.load(f)

st.write(
    """
    ____
   ## Adjust the Threshold Below to see the results 
   
"""
)

# Need a slider 
threshold_slider = st.slider("Probability Threshold", 0.0, 1.0, 0.13)

# Take input from threshold_slider and get predictions 
y_pred = x_prob_results[:, 1] > threshold_slider


with open("models/y_results.npy", "rb") as f:
    y_results = np.load(f)


with open("models/X_tra_results.npy", "rb") as f:
    X_transaction = np.load(f)


precision, recall, thresholds = precision_recall_curve(
    y_results, x_prob_results[:, 1]
)
auc_curve = metrics.auc(recall, precision)

confusion_matric_df = pd.DataFrame(metrics.confusion_matrix(y_results, y_pred))


st.write(
    """
  
   ### Precision & Recall for the specified threshold rate
   ____
"""
)

# Precision & Recall

plt.rcParams["font.family"] = "DIN Alternate"
fig1, ax = plt.subplots(figsize=[6.5, 6])
ax.grid(False)
ax.set_facecolor("1")
ax.spines["bottom"].set_color("0")
ax.spines["left"].set_color("0")
plt.title("RandomForest", fontweight="bold")
plt.plot(thresholds, precision[:-1], "--", label="Precision", color="r")
plt.plot(thresholds, recall[:-1], "--", label="Recall", color="b")
plt.xlabel("Threshold", fontweight="bold")
plt.legend(loc="lower left")
plt.ylim([0, 1])
ax.axvline(x=thresh, ymin=0, ymax=1, color="g")
st.pyplot(fig1)




fraud_recall = recall_score(y_results, y_pred)
nfraud_recall = recall_score(y_results, y_pred, pos_label=0)


savings = {
    "Prediction": y_pred.tolist(),
    "Actual": y_results.tolist(),
    "Transaction": X_transaction.tolist(),
}



st.write(
    """
____

### Confusion Matrix for the specified threshold rate
____

"""
)


# Confusion Matrix

confusion_matric_df = pd.DataFrame(metrics.confusion_matrix(y_results, y_pred))

fig2, ax = plt.subplots()
sns.set(font_scale=1.4)

sns.heatmap(
    confusion_matric_df,
    annot=True,
    fmt=".0f",
    annot_kws={"size": 14},
    xticklabels=["Non-Fraud", "Fraud"],
    yticklabels=["Non-Fraud", "Fraud"],
    cmap="gray_r",
)
plt.yticks(rotation=0)
plt.xlabel("Predicted", fontweight="bold")
plt.ylabel("Actual", rotation=0, y=0.92, fontweight="bold")
plt.title("Prediction Results", fontweight="bold", size=20)
st.pyplot(fig2)


st.write(
    """
    ____
   
"""
)
url_link = "https://github.com/kasteway/Fraud_Transaction"
st.markdown(url_link)

st.write(
    """
    ____
   
"""
)
