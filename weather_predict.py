#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import numpy as np
import pandas as pd 
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

st.title("Dự báo thời thiết :fog: :snowman: :umbrella: :lightning:")
st.markdown("Dự báo thời tiết sử dụng các thuật toán trong học máy không giám sát là đề tài đồ án cuối kỳ của môn học **Phương pháp nghiên cứu liên ngành**. Dưới đây là kết quả nghiên cứu và tìm hiểu của nhóm.")
st.divider()
st.subheader("Data collection")

df=pd.read_csv("https://raw.githubusercontent.com/thuuel/PPNCLN/main/weather_data.csv")
df.set_index('Date').sort_index()

columns_of_interest = ['TempAvgF','DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH', 'PrecipitationSumInches']
data = df[columns_of_interest]
events = df[['Events']].replace(' ', 'None')

st.markdown('Đây là bộ dữ liệu nhóm thu thập được và chưa qua các bước xử lý')
st.dataframe(data=data.head(), width=None, height=None)
st.dataframe(data=events.head(), width=None, height=None)

# Vẽ đồ thị thể hiện số lượng các sự kiện thời tiết đã xảy ra trong bộ dữ liệu
fig, ax = plt.subplots(figsize=(8,4))
ax1 = events.Events.value_counts().plot(kind='bar', color = plt.cm.Set2(range(len(events.Events.unique()))), ax=ax)
ax1.set_title("Weather events in dataset", fontsize=18)

# Lọc ra những sự kiện thời tiết đơn lẻ
unique_events = set()
for value in events.Events.value_counts().index:
    splitted = [x.strip() for x in value.split(',')]
    unique_events.update(splitted)

# Gán giá trị True cho các sự kiện thời tiết xảy ra tương ứng với từng hàng, ngược lại thì False
single_events = pd.DataFrame()
for event_type in unique_events:
    event_occurred = events.Events.str.contains(event_type)
    single_events = pd.concat([single_events, pd.DataFrame(data={event_type: event_occurred.values})], join='outer', axis=1)

# Vẽ đồ thị thể hiện số lượng của từng sự kiện thời tiết riêng lẻ trong bộ dữ liệu
fig, ax = plt.subplots(figsize=(8,4))
single_events.sum().sort_values(ascending=False).plot.bar(color = plt.cm.Set2(range(len(events.Events.unique()))), ax=ax)
ax.set_title("Weather events in dataset", fontsize=18)
ax.set_ylabel("Number of occurrences", fontsize=14)
for i in ax.patches:
    ax.text(i.get_x()+.18, i.get_height()+5, i.get_height(), fontsize=10)

# # Trong bộ dữ liệu đang sử dụng, có thể thấy ở cột PrecipitationSumInches có các giá trị T bên cạnh những số cụ thể.
# # Điều này có thể hiểu là vào ngày hôm đó có mưa nhưng không biết cụ thể là bao nhiêu.
precipitation = data[pd.to_numeric(data.PrecipitationSumInches, errors='coerce').isnull()].PrecipitationSumInches.value_counts()

# # Kiểm tra xem bộ dữ liệu được sử dụng có bao nhiêu hàng không phải là số.
def isColumnNotNumeric(columns_of_interest, data):
    result = np.zeros(data.shape[0], dtype=bool)
    for column_name in columns_of_interest:
        result = result | pd.to_numeric(data[column_name], errors='coerce').isnull()
    return result
def getDataFrameWithNonNumericRows(dataFrame):
    return data[isColumnNotNumeric(columns_of_interest, data)]

non_numeric_rows_count = getDataFrameWithNonNumericRows(data).shape[0]
print("Non numeric rows: {0}".format(non_numeric_rows_count))


# Chuyển đổi các dòng có T trong cột PrecipitationSumInches thành số 0.
# Đồng thời, tạo một cột mới tên PrecipitationTrace để lưu trữ các giả trị T này (gán 1 cho những dòng có T, 0 cho những dòng còn lại)
def numberOrZero(value):
    try:
        parsed = float(value)
        return parsed
    except:
        return 0

#Find rows indices with "T" values
has_precipitation_trace_series = isColumnNotNumeric(['PrecipitationSumInches'], data).astype(int)
data = data.assign(PrecipitationTrace=has_precipitation_trace_series.values)
data['PrecipitationSumInches'] = data['PrecipitationSumInches'].apply(numberOrZero)

# Từ các output trên, có thể thấy, ngoài cột T thì có nhiều cột khác chứa giá trị không phải số và đã được chuyển thành null.
# Vì thế, phải cân nhắc xử lí để mô hình chạy hiệu quả.
getDataFrameWithNonNumericRows(data)

row_indices_for_missing_values = getDataFrameWithNonNumericRows(data).index.values
data_prepared = data.drop(row_indices_for_missing_values)
events_prepared = single_events.drop(row_indices_for_missing_values)
print("Data rows: {0}, Events rows: {1}".format(data_prepared.shape[0], events_prepared.shape[0]))

# Vì mô hình của học máy không giám sát nhóm sử dụng được chạy trên các dữ liệu dạng số.
# Vì thế, phải kiểm tra xem loại dữ liệu của các cột và ép kiểu nếu cần.
# data_prepared.dtypes
data_prepared = data_prepared.apply(pd.to_numeric)
# Bắt đầu chuẩn hóa dữ liệu để huấn luyện mô hình

from sklearn import preprocessing
data_values = data_prepared.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
data_prepared = pd.DataFrame(min_max_scaler.fit_transform(data_prepared), columns=data_prepared.columns, index=data_prepared.index)

st.divider()
st.subheader("Data Preparation")
st.markdown('Sau khi được làm sạch, xử lý và chuẩn hóa, đây là bộ dữ liệu cuối cùng nhóm dùng để huấn luyện và kiểm thử')
st.dataframe(data=data_prepared.head(), width=None, height=None)
st.dataframe(data=events_prepared.head(), width=None, height=None)

# Chia dữ liệu thành 2 tập riêng biệt để huấn luyện và kiểm thử
from sklearn.model_selection import train_test_split
random_state = 42
X_train, X_test = train_test_split(data_prepared, test_size=0.2, random_state=random_state)
y_train, y_test = train_test_split(events_prepared, test_size=0.2, random_state=random_state)

clusters_count = len(unique_events)

# Sử dụng những thuật toán phân cụm và so sánh với kết quả thực tế. Từ đó, đưa ra thuật toán cho kết quả gần với thực tế nhất.
st.divider()
st.subheader("Xây dựng mô hình phân cụm")
st.markdown('Đây là mô hình phân cụm sử dụng **thuật toán K-Means**')
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
kmeans = KMeans(n_clusters=clusters_count).fit(X_train)
resultDf1 = pd.DataFrame(kmeans.labels_)
fig, ax = plt.subplots(figsize=(8,4))
ax1 = resultDf1.iloc[:,0].value_counts().plot.bar(color = plt.cm.Set2(range(len(events.Events.unique()))), ax=ax)
ax1.set_title("K-Means clustering", fontsize=16)
st.pyplot(fig)

st.markdown('Đây là mô hình phân cụm sử dụng **thuật toán Spectral Clustering**')
from sklearn.cluster import SpectralClustering
warnings.filterwarnings("ignore")
sc = SpectralClustering(n_clusters=clusters_count).fit(X_train)
resultDf2 = pd.DataFrame(sc.labels_)
fig, ax = plt.subplots(figsize=(8,4))
ax1 = resultDf2.iloc[:,0].value_counts().plot.bar(color = plt.cm.Set2(range(len(events.Events.unique()))), ax=ax)
ax1.set_title("Spectral Clustering", fontsize=16)
st.pyplot(fig)

st.markdown('Đây là mô hình phân cụm sử dụng **thuật toán DBSCAN**')
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.25, min_samples=4).fit(X_train)
resultDf3 = pd.DataFrame(dbscan.labels_)
fig, ax = plt.subplots(figsize=(8,4))
ax1 = resultDf3.iloc[:,0].value_counts().plot.bar(color = plt.cm.Set2(range(len(events.Events.unique()))), ax=ax)
ax1.set_title("DBSCAN", fontsize=16)
st.pyplot(fig)

st.markdown('Đây là mô hình phân cụm sử dụng **thuật toán Agglomerative Clustering**')
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=clusters_count, linkage="average").fit(X_train)
resultDf = pd.DataFrame(ac.labels_)
fig, ax = plt.subplots(figsize=(8,4))
ax1 = resultDf.iloc[:,0].value_counts().plot.bar(color = plt.cm.Set2(range(len(events.Events.unique()))), ax=ax)
ax1.set_title("Agglomerative Clustering", fontsize=16)
st.pyplot(fig)

st.markdown('Đây là đồ thị thể hiện số **sự kiện đơn lẻ** xảy ra trong thực tế')
fig, ax = plt.subplots(figsize=(8,4))
ax1 = events_prepared.sum().sort_values(ascending=False).plot.bar(color = plt.cm.Set2(range(len(events.Events.unique()))), ax=ax)
ax1.set_title("Single weather events in dataset", fontsize=18)
st.pyplot(fig)
st.divider()
st.subheader("Đánh giá và lựa chọn thuật toán")
st.markdown('Sau khi cân nhắc và so sánh kết quả phân cụm của 4 thuật toán, có thể thấy, thuật toán **Agglomerative Clustering** đem lại kết quả phân cụm gần với thực tế nhất.')
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
events_prepared.sum().sort_values(ascending=False).plot.bar(ax=ax[0], title="Real events that happened", color = plt.cm.Set2(range(len(events.Events.unique()))))
resultDf.iloc[:,0].value_counts().plot.bar(ax=ax[1], title="Bar obtained from agglomerative clustering", color = plt.cm.Set2(range(len(events.Events.unique()))))
st.pyplot(fig)

# Thực hiện gán tên cụm vào tên sự kiện thời tiết tương ứng. 
# Sau đó, sử dụng lý thuyết phân cụm của thuật toán Agglomerative để xem rằng liệu 1 ngày có thể có 2 sự kiện thời tiết hay không.
st.divider()
st.subheader("Ứng dụng và đánh giá kết quả phân cụm")
st.markdown("Từ kết quả phân cụm ở trên, cùng với các cơ sở lý thuyết phân cụm bằng Agglometive, nhóm tiến hành phân cụm lại các điểm để xem rằng liệu có điểm nào có thể thuộc vào 2 cụm riêng biệt không, nghĩa là cùng một thông số thời tiết thì liệu có thể xảy ra từ 2 sự kiện trở lên không.")
event_names_ordered = events_prepared.sum().sort_values(ascending=False).index
clusters_ordered = resultDf.iloc[:,0].value_counts().index

cluster_category_mapping = {}
for i in range(clusters_count):
    # Chuyển đổi numpy.float64 thành float
    key = int(clusters_ordered[i])
    value = str(event_names_ordered[i])
    cluster_category_mapping.update({key: value})

    
cluster_centers_mapping = {}
for key in cluster_category_mapping:
    cluster_indices = resultDf.loc[resultDf[0] == key].index
    cluster_data = X_train.iloc[cluster_indices]
    mean = cluster_data.mean(axis=0).values
    cluster_centers_mapping.update({key:mean})
    
#tính khoảng cách từ 1 điểm đến tâm từng cụm
def get_distances_from_cluster(data_frame):
    cluster_distance = np.zeros((data_frame.shape[0], clusters_count))
    #khoảng cách euclidean
    for i in range(data_frame.shape[0]):
        for key in cluster_category_mapping:
            dist = np.linalg.norm(data_frame.iloc[[i]].values[0]-cluster_centers_mapping[key])
            cluster_distance[i,key] = dist
            #print(dist)
    column_names = [cluster_category_mapping[k] for k in cluster_category_mapping]
    return pd.DataFrame(cluster_distance, index=data_frame.index, columns=column_names)

distancesDf = get_distances_from_cluster(X_train)

def classify_events(distances_dataFrame):
    return distances_dataFrame.apply(lambda x: x<x.min()*1.01, axis=1)

classification_result = classify_events(distancesDf)
X_train_col_ordered = classification_result.reindex(sorted(classification_result.columns), axis=1)
y_train_col_ordered = y_train.reindex(sorted(y_train.columns), axis=1)

#xem giá trị  dự đoán có đúng vs gtri thực tế
def check_accuracy(X, y):    
    comparison = X == y
    val_counts = comparison.all(axis=1).value_counts()
    percentageCorrect = val_counts.at[True] / X.shape[0] * 100
    return percentageCorrect


# Đánh giá mô hình phân cụm dựa trên tỷ lệ giữa số dòng dự báo đúng và tổng số dòng của 2 tập X_train và X_test

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
y_train_col_ordered.sum().plot.bar(ax=ax[0], title="Real events that happened", color = plt.cm.Set2(range(len(events.Events.unique()))))
ax = X_train_col_ordered.sum().plot.bar(ax=ax[1], title="Predicted events", color = plt.cm.Set2(range(len(events.Events.unique()))))
#resultDf.iloc[:,0].value_counts().plot.bar(ax=ax[1], title="Histogram obtained from agglomerative clustering")
st.pyplot(fig)
a = check_accuracy(X_train_col_ordered, y_train_col_ordered)
st.write(f'Chỉ số Accuracy của mô hình ở tập X_train là {a}')

distancesDf = get_distances_from_cluster(X_test)
classification_result = classify_events(distancesDf)
X_test_col_ordered = classification_result.reindex(sorted(classification_result.columns), axis=1)
y_test_col_ordered = y_test.reindex(sorted(y_train.columns), axis=1)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
y_test_col_ordered.sum().plot.bar(ax=ax[0], title="Real events that happened", color = plt.cm.Set2(range(len(events.Events.unique()))))
X_test_col_ordered.sum().plot.bar(ax=ax[1], title="Predicted events", color = plt.cm.Set2(range(len(events.Events.unique()))))
st.pyplot(fig)
a = check_accuracy(X_test_col_ordered, y_test_col_ordered)
st.write(f'Chỉ số Accuracy của mô hình ở tập X_test là {a}')
import joblib
joblib.dump(min_max_scaler, 'min_max_scaler.joblib')
min_max_scaler = joblib.load('min_max_scaler.joblib')

with st.sidebar:
    st.subheader("Vui lòng nhập các thông số dưới đây!")
    temp_text = st.number_input("Nhiệt độ trung bình (độ F)") 
    dewpoint_text = st.number_input("Nhiệt độ điểm sương trung bình (độ F)") 
    humidty_text = st.number_input("Độ ẩm trung bình (%)")
    slp_text = st.number_input("Áp suất trung bình ở mực nước biển (inHg)")
    visibility_text = st.number_input("Tầm nhìn trung bình (m)")
    windspeed_text = st.number_input("Tốc độ gió trung bình (miles/h)")
    precipi_text = st.number_input("Tổng lượng mưa (inch)")
    t = 0
    data_input = {
    'TempAvgF': [temp_text],
    'DewPointAvgF': [dewpoint_text],
    'HumidityAvgPercent': [humidty_text],
    'SeaLevelPressureAvgInches': [slp_text],
    'VisibilityAvgMiles': [visibility_text],
    'WindAvgMPH': [windspeed_text],
    'PrecipitationSumInches': [precipi_text],
    'PrecipitationTrace': [t]}
    input_df = pd.DataFrame(data_input)
    st.dataframe(data=input_df, width=None, height=None)

    input_df = pd.DataFrame(min_max_scaler.fit_transform(input_df), columns=input_df.columns, index=input_df.index)
    st.dataframe(data=input_df, width=None, height=None)
    button = st.button('Predict')
    if button:
        distancedf_input = get_distances_from_cluster(input_df)
        st.dataframe(data=distancedf_input, width=None, height=None)
        result_events = classify_events(distancedf_input)
        st.dataframe(data=result_events, width=None, height=None)
        true_columns = result_events.apply(lambda row: row.index[row].tolist(), axis=1)
        st.write(f"{temp_text}, {dewpoint_text}, {humidty_text} chúng tôi dự đoán rằng hôm ấy trời có {true_columns.tolist()}")
        
       
