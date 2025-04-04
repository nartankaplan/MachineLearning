# cap-diameter ve stem-height arasında scatter plot
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x='cap-diameter', y='stem-height', hue='class', palette='Set1')
plt.title("cap-diameter vs stem-height (Colored by Class)")
plt.show()
