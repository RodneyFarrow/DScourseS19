df1 <- iris 

df <- createDataFrame(iris)

df2 <- iris

print(class(df1))

print(class(df))

#print(head(select(df, df$Sepal_Length, df$Species), 6))

#print(head(filter(df, df$Sepal_Length>5.5)))

#print(head(filter(df1, df$Sepal_Length>5.5)))

#print(head(select(df1, df$Sepal_Length, df$Species), 6))

print(head(select(filter(df, df$Sepal_Length>5.5), df$Sepal_Length, df$Species)))

head(summarize(groupBy(df, df$Species),mean=mean(df$Sepal_Length), count=n(df$Sepal_Length)))

head(arrange(df, df$Species),mean=mean(df$Sepal_Length), count=n(df$Sepal_Length))

#head(arrange(df2, asc(df2$Species)))
