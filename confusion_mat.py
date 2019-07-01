import seaborn as sn
cm_df = pd.DataFrame(confusion_matrix(y_test_nocat, y_pred), columns=['Normal','Crackles','Wheezes'])
cm_df = cm_df.rename(index={0:'Normal',1:'Crackles',2:'Wheezes'})
ax = sn.heatmap(cm_df, annot=True, fmt='g', cmap="Blues", annot_kws={"size": 20})
ax.set_xticklabels(bars,size = 15)
ax.set_yticklabels(bars,size = 15, rotation = 30)
plt.xlabel('Predicted', size = 20)
plt.ylabel('True', size = 20)