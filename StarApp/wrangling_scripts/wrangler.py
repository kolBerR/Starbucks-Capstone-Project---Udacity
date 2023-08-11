import numpy as np
import pandas as pd
import plotly.graph_objs as go

from models.model import apply_model

# read in the json files containing the datasets
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records',lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)


def calculate_offer_success():
    """ 
    A function to calculate the success percentage for each offer
    
    INPUT:
        clean_data (dataframe): DataFrame that includes offer_completed along with other offer information
        portfolio (dataframe): The original dataset containing offer_types, duration, etc
    
    OUTPUT:
        percent_success (dataframe): DataFrame that describes the success percentage for each offer
    """
    # read the wrangled csv data from the data folder
    final_data= pd.read_csv('data/final_data.csv')
    # group the completed offers by offer_id and aggregate
    success_count = final_data[['offer_id', 'offer_completed']].groupby('offer_id').sum().reset_index()
    # value counts for the offer_ids
    offer_count = final_data['offer_id'].value_counts()
    # generate a Pandas dataframe from the indexes and value of the offer_count above
    offer_count = pd.DataFrame(list(zip(offer_count.index.values, offer_count.values)), columns=['offer_id', 'count'])

    # sort the success count data generated above
    sorted_success_count = success_count.sort_values('offer_id')
    # sort the offer_count data
    sorted_offer_count = offer_count.sort_values('offer_id')
    # merge data contained in offer_count and success_count
    offer_success_count = pd.merge(sorted_offer_count, sorted_success_count, on="offer_id")

    # create a success_percent column by dividing offer completed by offer count
    offer_success_count['success_percent'] = (100 * offer_success_count['offer_completed'] / offer_success_count['count'])

    # create a copy of the original porfolio dataset with only the 'id' and 'offer_type' columns included
    portfolio_copy = portfolio[['id', 'offer_type']]
    # rename the 'id' column label to 'offer_id' to facilitate a merge function
    portfolio_copy = portfolio_copy.rename(columns = {'id': 'offer_id'})
    # merge the offer_success_count and portfolio dataframes
    merged_count = pd.merge(offer_success_count, portfolio_copy, on="offer_id")

    # drop the offer_completed column since the desired information is now captured in the new 'success_percent' column
    percent_success = merged_count.drop(columns=['offer_completed'])
    # sort the success percentages 
    percent_success = percent_success.sort_values('success_percent', ascending=False).reset_index(drop=True)

    return percent_success



def return_figures():
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

  # first chart plots arable land from 1990 to 2015 in top 10 economies 
  # as a line chart
    
    graph_one = []
    df = portfolio.groupby('offer_type')['id'].count()
    
    x_val = df.index.tolist()
    y_val =  df.tolist()
    graph_one.append(
        go.Bar(
        x = x_val,
        y = y_val,
        marker_color='rgb(242, 140, 40)'
        )
    )

    layout_one = dict(title = 'Summary of Offers',
                xaxis = dict(title = 'Offer Type'),
                yaxis = dict(title = 'Number of Offers'),
                )

# second chart plots ararble land for 2015 as a bar chart    
    graph_two = []

    df = profile[['age', 'income']]
    bins = pd.cut(df['age'], bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]).astype(str)
    age_income = df.groupby(bins)['income'].agg(['mean']).round(2)

    labels = ['20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s', '100+']
    vals = np.reshape(age_income, (1,-1))[0].tolist()
    income_age_group = pd.Series(vals, index=labels)

    x_val = income_age_group.index.tolist()
    y_val = income_age_group.values.tolist()

    graph_two.append(
      go.Bar(
      x = x_val,
      y = y_val,
      marker_color = 'rgb(156,175,136)',
      marker_line_color='rgb(8,48,107)'
      )
    )

    layout_two = dict(title = 'Age-Income Distrubtion of Customers',
                xaxis = dict(title = 'Age Interval',),
                yaxis = dict(title = 'Average Income in USD'),
                )


# third chart plots percent of population that is rural from 1990 to 2015
    graph_three = []
    df = round(transcript['event'].value_counts()/len(transcript['event']),2)
 
    x_val = df.index.tolist()
    y_val =  df.tolist()
    graph_three.append(
        go.Scatter(
        x = x_val,
        y = y_val,

        )
    )

    layout_three = dict(title = 'Proportion of Offer Types in Transcript',
                xaxis = dict(title = 'Event'),
                yaxis = dict(title = 'Percent of Offer Types'),
                )
    
# fourth chart shows rural population vs arable land
    graph_four = []
    
    df = calculate_offer_success()

    x_val = df.index.tolist()
    y_val = df['success_percent'].values.tolist()


    graph_four.append(
        go.Bar(
        x = x_val,
        y = y_val,
    
        )
    )

    layout_four = dict(title = 'Percentage of Offers Accepted by Customers',
                xaxis = dict(title = 'Offer Number'),
                yaxis = dict(title = 'Success_Percent'),
                )
    
    
    graph_five = []

    #we could use the below line to get the data from models.py. However, to delay
    # the load time of our webpage due to data processing and model execusion, we
    # will directly use the results
    # matrix, x, y = apply_model()

    matrix = [[9458, 1042],
       [ 816, 8635]]
    x_vals = [57.65, 5.95, 4.6, 4.32, 4.09]
    y_features = ['total_amount', 'month', 'difficulty', 'duration_in_hrs', 'reward']


    graph_five.append(

        go.Table(
            header=dict(values=['Positive', 'Negative'], fill_color='seagreen'),
            cells=dict(values=[matrix[0], matrix[1]], fill_color='lightcyan'))
           
        )


    graph_six = []

    graph_six.append(
        go.Bar(
        x = x_vals,
        y = y_features,
        orientation='h',
        marker_color='rgb(128, 0, 128)'
        )
    )
    
    layout_six = dict(title = 'Top 5 features with the strongest explanatory power',
                xaxis = dict(title = 'Feature'),
                yaxis = dict(title = 'Percentage'),
                
                )
    
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))
    figures.append(dict(data=graph_five))
    figures.append(dict(data=graph_six, layout=layout_six))

    return figures

