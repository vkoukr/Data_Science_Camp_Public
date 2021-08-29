#----Load library
import numpy as np
import pandas as pd

#----Load data
from sklearn.datasets import load_breast_cancer
breast = load_breast_cancer()
breast_data = breast.data
print(breast_data.shape)

breast_labels = breast.target
print(breast_labels.shape)

labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)
print(final_breast_data.shape)

#----Create Dataframe
breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
features_labels = np.append(features,'label')
breast_dataset.columns = features_labels
breast_dataset.head()

#-----PCA
#----Applying Scaler
from sklearn.preprocessing import StandardScaler
x = breast_dataset.loc[:, features].values
x_scaled = StandardScaler().fit_transform(x) # normalizing the features
x_scaled.shape
np.mean(x_scaled),np.std(x_scaled)

feat_cols = ['feature'+str(i) for i in range(x_scaled.shape[1])]
normalised_breast = pd.DataFrame(x_scaled,columns=feat_cols)

from sklearn.decomposition import PCA
pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x_scaled)
principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2'])

#------Plotting
import matplotlib.pyplot as plt
import seaborn as sns

ax = plt.figure(figsize=(12,8))
sns.scatterplot(principalComponents_breast[:,0], principalComponents_breast[:,1], hue=breast_dataset['label'], palette ='Set1' )
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')