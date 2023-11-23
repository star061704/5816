import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from scipy import stats
from collections import Counter
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

data = pd.read_csv("abalone.data")
data['Sex'] = data['Sex'].map({'M': 0, 'F': 1, 'I': 2})

def categorize_ring_age(age):
    if 0 <= age <= 7:
        return 1
    elif 8 <= age <= 10:
        return 2
    elif 11 <= age <= 15:
        return 3
    elif age>15 :
        return 4



data['ring_age_group'] = data['Rings'].apply(categorize_ring_age)
print(Counter(data['ring_age_group']))

sns.countplot(x='ring_age_group',data = data)
plt.savefig('distribution of class')
plt.close()

# print(data)

sns.distplot(data['Diameter'])
plt. tight_layout ()
plt.savefig('Diameter')
plt.close()

sns.distplot(data['Length'])
plt. tight_layout ()
plt.savefig('Length')
plt.close()

sns.distplot(data['Height'])
plt. tight_layout ()
plt.savefig('Height')
plt.close()

sns.distplot(data['Whole_weight'])
plt. tight_layout ()
plt.savefig('Whole_weight')
plt.close()

sns.distplot(data['Shucked_weight'])
plt. tight_layout ()
plt.savefig('Shucked_weight')
plt.close()

sns.distplot(data['Viscera_weight'])
plt. tight_layout ()
plt.savefig('Viscera_weight')
plt.close()

sns.distplot(data['Shell_weight'])
plt. tight_layout ()
plt.savefig('Shell_weight')
plt.close()

sns.distplot(data['Rings'])
plt. tight_layout ()
plt.savefig('Rings')
plt.close()

data['Sex']=pd.to_numeric(data['Sex'],errors='coerce')
cor_matrix = data.corr().round(2)
sns.heatmap(data=cor_matrix,annot=True)
plt.tight_layout()
plt.savefig('correlation map.png')
plt.close()

