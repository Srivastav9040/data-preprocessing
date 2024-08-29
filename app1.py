import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import random
import plotly.express as px
import folium
from streamlit_folium import folium_static
import time
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from joblib import load

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

st.markdown(
    """
    <style>
        body {
            background-image: url('your_image_url.jpg');
            background-size: cover;
        }
    </style>
    """,
    unsafe_allow_html=True
)

a = st.sidebar.selectbox("Select what do you want to do",["Home",'Price prediction','Analysis','Recommender system','Squared area'])
if a == "Squared area":
   new_df = pd.read_csv('C:/Users/anura/OneDrive/Desktop/real-estate-app-master/datasets/data_viz1.csv')
   group_df = new_df.groupby('sector')[['price','price_per_sqft','built_up_area','latitude','longitude']]
   new_df1 = new_df.drop_duplicates()
   new_df = pd.DataFrame(new_df)


   st.title("Select Places to Display on Map")


   selected_sectors = st.multiselect("Select two sectors:", new_df1['sector'].unique().tolist())

   if len(selected_sectors) == 2:
     filtered_df = new_df[new_df['sector'].isin(selected_sectors)]

  
     map_center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
     m = folium.Map(location=map_center, zoom_start=12)

    # Add markers for selected sectors
     for index, row in filtered_df.iterrows():
        folium.Marker(location=[row['latitude'], row['longitude']], popup=row['sector']).add_to(m)

    # Display the map using Streamlit
     st.subheader(f"Map for selected sectors:")
     folium_static(m)
elif a == 'Price prediction':
    st.header('Enter your inputs')
    df = pd.read_csv('C:/Users/anura/OneDrive/Desktop/Capstone/ipynb/gurgaon_properties_post_feature_selection_v2.csv')

    df = df.head(500)
    property_type = st.selectbox('Property Type',['flat','house'])
    sector = st.selectbox('Sector',sorted(df['sector'].unique().tolist()))

    bedrooms = float(st.selectbox('Number of Bedroom',sorted(df['bedRoom'].unique().tolist())))

    bathroom = float(st.selectbox('Number of Bathrooms',sorted(df['bathroom'].unique().tolist())))

    balcony = st.selectbox('Balconies',sorted(df['balcony'].unique().tolist()))

    property_age = st.selectbox('Property Age',sorted(df['agePossession'].unique().tolist()))

    built_up_area = float(st.number_input('Built Up Area'))

    servant_room = float(st.selectbox('Servant Room',[0.0, 1.0]))
    store_room = float(st.selectbox('Store Room',[0.0, 1.0]))

    furnishing_type = st.selectbox('Furnishing Type',sorted(df['furnishing_type'].unique().tolist()))
    luxury_category = st.selectbox('Luxury Category',sorted(df['luxury_category'].unique().tolist()))
    floor_category = st.selectbox('Floor Category',sorted(df['floor_category'].unique().tolist()))

    #st.title("Predict Price App")
    columns_to_encode = ['property_type','sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
        ('cat', OrdinalEncoder(), columns_to_encode),
        ('cat1',OneHotEncoder(drop='first',sparse_output=False),['sector','agePossession'])
    ], 
    remainder='passthrough'
)
    
    pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=500))
])
    X = df.drop(columns=['price'])
    y = df['price']
    y_transformed = np.log1p(y)
    pipeline.fit(X,y_transformed)

    if st.button('Predict'):
      progress_bar = st.progress(0)  
      data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
      columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']
      one_df = pd.DataFrame(data, columns=columns)
      progress_bar.progress(20)
      base_price = np.expm1(pipeline.predict(one_df))[0]

      low = base_price - 0.22
      high = base_price + 0.22
      progress_bar.progress(50)
      st.header('Your inputs are')
      st.dataframe(one_df)
      import time
      time.sleep(2)
      progress_bar.progress(80)
      st.success("🏡 The price of the flat is between {} Cr and {} Cr".format(round(low, 2), round(high, 2)))
      progress_bar.progress(100)


    # Display the result with styled text
    #st.markdown("<p class='stText'>The price is between <span style='color: green;'>{}</span> and <span style='color: green;'>{}</span> crs</p>".format(rounded_number1, rounded_number), unsafe_allow_html=True)

elif a == 'Home':
  st.title('Welcome to the GurgaonDwellForecast')


elif a == 'Recommender system':
    df = pd.read_csv('C:/Users/anura/OneDrive/Desktop/real-estate-app-master/datasets/loc.csv')


    st.title('Select a location and radius')

    loca  = st.selectbox('Location',sorted(df.columns.to_list()))
    rad = st.number_input("Select Radius")


    if st.button('Search'):
      res = df[df[loca]<rad*1000].set_index('PropertyName')[loca].sort_values()
      for k,val in res.items():
        st.success("🗺️ {}    {}".format(k, val))

    
    df1 = pd.read_csv('C:/Users/anura/OneDrive/Desktop/real-estate-app-master/datasets/loc.csv')
    df1 = df1.set_index('PropertyName')
    st.title('Recommended Appartments')
    selected_appartment = st.selectbox('Select an appartment',sorted(df1.index.to_list()))
    cosine_sim1 = pickle.load(open('C:/Users/anura/OneDrive/Desktop/real-estate-app-master/datasets/cosine_sim1.pkl','rb'))
    cosine_sim2 = pickle.load(open('C:/Users/anura/OneDrive/Desktop/real-estate-app-master/datasets/cosine_sim2.pkl','rb'))
    cosine_sim3 = pickle.load(open('C:/Users/anura/OneDrive/Desktop/real-estate-app-master/datasets/cosine_sim3.pkl','rb'))
    def recommend_properties_with_scores(property_name, top_n=5):
      cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1 * cosine_sim3
      sim_scores = list(enumerate(cosine_sim_matrix[df1.index.get_loc(property_name)]))
      sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
      top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
      top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]
      top_properties = df1.index[top_indices].tolist()
      recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'Similarity': top_scores
       })
      return recommendations_df
    if st.button('Recommend'):
      df = recommend_properties_with_scores(selected_appartment)
      #st.table(df)
      st.write(df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'))


else:
   new_df = pd.read_csv('C:/Users/anura/OneDrive/Desktop/real-estate-app-master/datasets/data_viz1.csv')
   group_df = new_df.groupby('sector')[['price','price_per_sqft','built_up_area','latitude','longitude']]
   file_path = "C:/Users/anura/OneDrive/Desktop/real-estate-app-master/datasets/data_viz1.csv"  
   nrows = st.slider("Zoom it out", min_value=10, max_value=1000, value=100)
   df = pd.read_csv(file_path, nrows=nrows)
   map_center = [df['latitude'].iloc[0], df['longitude'].iloc[0]]
   m = folium.Map(location=map_center, zoom_start=12)
   for index, row in df.iterrows():
        folium.Marker(location=[row['latitude'], row['longitude']], popup=row['sector']).add_to(m)
   folium_static(m)

   

   st.header('Area Vs Price')

   property_type = st.selectbox('Select Property Type', ['flat','house'])

   if property_type == 'house':
     fig1 = px.scatter(new_df[new_df['property_type'] == 'house'], x="built_up_area", y="price", color="bedRoom", title="Area Vs Price")

     st.plotly_chart(fig1, use_container_width=True)
   else:
     fig1 = px.scatter(new_df[new_df['property_type'] == 'flat'], x="built_up_area", y="price", color="bedRoom",
                      title="Area Vs Price")

     st.plotly_chart(fig1, use_container_width=True)

   st.header('BHK Pie Chart')

   sector_options = new_df['sector'].unique().tolist()
   sector_options.insert(0,'overall')

   selected_sector = st.selectbox('Select Sector', sector_options)

   if selected_sector == 'overall':

     fig2 = px.pie(new_df, names='bedRoom')

     st.plotly_chart(fig2, use_container_width=True)
   else:

     fig2 = px.pie(new_df[new_df['sector'] == selected_sector], names='bedRoom')

     st.plotly_chart(fig2, use_container_width=True)

    

   st.header('Side by Side BHK price comparison')

   fig3 = px.box(new_df[new_df['bedRoom'] <= 4], x='bedRoom', y='price', title='BHK Price Range')

   st.plotly_chart(fig3, use_container_width=True)


   st.header('Side by Side Distplot for property type')

   fig3 = plt.figure(figsize=(10, 4))
   sns.distplot(new_df[new_df['property_type'] == 'house']['price'],label='house')
   sns.distplot(new_df[new_df['property_type'] == 'flat']['price'], label='flat')
   plt.legend()
   st.pyplot(fig3)


   