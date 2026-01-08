#Assignment-2
# an Application Programming Interface or API
import matplotlib.pyplot as plt #import lib for plot
import csv #import lib for reading file

def get_city_temperatures(filename,city_name):
    """
    Extract temperature data for a specific city from CSV file.
    
    Parameters:
    filename (str): Path to the CSV file
    city_name (str): Name of the city to extract data for
    
    Returns:
    dict: Dictionary mapping 'YYYY-MM' to temperature (float)
          Returns empty dict if city not found
    """
    temperature_data = {}
    
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            # Check if this row matches our city
            if row['City'] == city_name:
                # Extract year-month from date (format: 1849-01-01 -> 1849-01)
                date_str = row['dt']
                year_month = date_str[:7]  # Take first 7 characters (YYYY-MM)
                
                # Get temperature, handle missing values
                temp_str = row['AverageTemperature']
                if temp_str and temp_str.strip():  # Check if not empty
                    try:
                        temperature = float(temp_str)
                        temperature_data[year_month] = temperature
                    except ValueError:
                        # Skip rows with invalid temperature data
                        continue
        
        x=list(temperature_data.keys())#take list of x - 'year_month'
        y=list(temperature_data.values())#take list of y-corresponding 'temperature'
        plt.figure(figsize=(16,6))
        plt.plot(x, y, marker="o", linestyle="-", color="green")#datapoints makrked as 'o', line connecting successive points
        plt.xticks(ticks=range(0, len(x), 35), labels=[x[i] for i in range(0, len(x), 35)],rotation=45)   # Rotate labels for readability
        plt.xlabel("Year-Month")
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature Trend")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return temperature_data


def get_available_cities(filename, limit=None):
    """
    Get list of unique cities in the dataset.
    
    Parameters:
    filename (str): Path to the CSV file
    limit (int): Maximum number of cities to return (None for all)
    
    Returns:
    list: List of unique city names
    """
    cities = set()
    
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            cities.add(row['City'])
            if limit and len(cities) >= limit:
                break
    
    return sorted(list(cities))

# =============================================================================
# ASSIGNMENT: Build a Temperature Data API
# =============================================================================

def find_temperature_extremes(filename, city_name):
    """
    Find the hottest and coldest months on record for a city.
    
    Parameters:
    filename (str): Path to the CSV file
    city_name (str): Name of the city
    
    Returns:
    dict: {
        'hottest': {'date': 'YYYY-MM', 'temperature': float},
        'coldest': {'date': 'YYYY-MM', 'temperature': float}
    }"""
    hottest = {"date": None, "temperature": float("-inf")}#setting hottest initially to negative infinity so that any temperature>-inf and can be first initialized as the hottest
    coldest = {"date": None, "temperature": float("inf")}#setting coldest initially to positive infinity so that any temperature<inf and can be first initialized as the coldest
    
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["City"] == city_name:
                temp_str = row["AverageTemperature"]
                if temp_str and temp_str.strip():
                    try:
                        temp = float(temp_str)
                        date_str = row["dt"][:7]  # YYYY-MM
                        if temp > hottest["temperature"]:#check if higher than current hottest
                            #change hottest
                            hottest = {"date": date_str, "temperature": temp}
                        if temp < coldest["temperature"]:#check if lower than current coldest
                            #change coldest
                            coldest = {"date": date_str, "temperature": temp}

                    except ValueError:
                        continue

    return {"hottest": hottest, "coldest": coldest}


def get_seasonal_averages(filename, city_name, season):
    """
    Calculate average temperature for a specific season across all years.
    Never mind that Chennai only has Hot, Hotter and Hottest...
    
    Parameters:
    filename (str): Path to the CSV file
    city_name (str): Name of the city
    season (str): 'spring', 'summer', 'fall', or 'winter'
    
    Returns:
    dict: {
        'city': str,
        'season': str,
        'average_temperature': float
    }
        
    Assume: Spring = Mar,Apr,May; Summer = Jun,Jul,Aug; 
          Fall = Sep,Oct,Nov; Winter = Dec,Jan,Feb
    """
    
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        summer_month_count,winter_month_count,spring_month_count,fall_month_count=0,0,0,0
        spring_temp_sum,summer_temp_sum,winter_temp_sum,fall_temp_sum=0,0,0,0
        for row in reader:
            if row["City"] == city_name:
                temp_str = row["AverageTemperature"]
                if temp_str and temp_str.strip():
                    try:
                        temp = float(temp_str)
                        year = int(row["dt"][:4])  # YYYY
                        month=int(row["dt"][5:7]) #MM
                        if month in (3,4,5):#check if spring
                            spring_month_count+=1
                            spring_temp_sum+=temp
                        elif month in (6,7,8):#check if summer
                            summer_month_count+=1
                            summer_temp_sum+=temp
                        elif month in (9,10,11):#check if fall
                            fall_month_count+=1
                            fall_temp_sum+=temp
                        elif month in (12,1,2):#check if winter
                            winter_month_count+=1
                            winter_temp_sum+=temp
                    except ValueError:
                        continue
        #calculate average temperature of the season by (temperature sum)/(temperature data count)
        if season=='summer':average_temperature=float(summer_temp_sum/summer_month_count)
        elif season=='winter':average_temperature=float(winter_temp_sum/winter_month_count)
        elif season=='spring': average_temperature=float(spring_temp_sum/spring_month_count)
        elif season=='fall': average_temperature=float(fall_temp_sum/fall_temp_sum)
        else: return None

    return {
        'city': city_name,
        'season': season,
        'average_temperature': average_temperature
    }


def compare_decades(filename, city_name, decade1, decade2):
    """
    Compare average temperatures between two decades for a city.
    
    Parameters:
    filename (str): Path to the CSV file
    city_name (str): Name of the city
    decade1 (int): First decade (e.g., 1980 for 1980s)
    decade2 (int): Second decade (e.g., 2000 for 2000s)
    
    Returns:
    dict: {
        'city': str,
        'decade1': {'period': '1980s', 'avg_temp': float, 'data_points': int},
        'decade2': {'period': '2000s', 'avg_temp': float, 'data_points': int},
        'difference': float,
        'trend': str  # 'warming', 'cooling', or 'stable'
    }
    """   
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        decade1_temp_count,decade1_temp_sum,decade2_temp_count,decade2_temp_sum=0,0,0,0,
        for row in reader:
            if row["City"] == city_name:
                temp_str = row["AverageTemperature"]
                if temp_str and temp_str.strip():
                    try:
                        temp = float(temp_str)
                        year = int(row["dt"][:4])  # YYYY
                        if decade1 <= year < decade1+10: #check if year falls in decade1
                            decade1_temp_count+=1
                            decade1_temp_sum+=temp
                        if decade2 <= year <= decade2+10: #check if year falls in decade2
                            decade2_temp_count+=1
                            decade2_temp_sum+=temp

                    except ValueError:
                        continue
        #calculate average temperature of the decade
        decade1_avg_temp=float(decade1_temp_sum/decade1_temp_count)
        decade2_avg_temp=float(decade2_temp_sum/decade2_temp_count)
        #compare average temperatures and find the temperature trend
        if decade1_avg_temp>decade2_avg_temp: trend='cooling'
        elif decade1_avg_temp<decade2_avg_temp: trend='warming'
        elif decade1_avg_temp==decade2_avg_temp: trend='stable'
    
    return{
        'city': city_name,
        'decade1': {'period': '1980s', 'avg_temp': decade1_avg_temp, 'data_points': decade1_temp_count},
        'decade2': {'period': '2000s', 'avg_temp': decade2_avg_temp, 'data_points': decade2_temp_count},
        'difference': abs(decade1_avg_temp-decade2_avg_temp),
        'trend': trend
    }

def find_similar_cities(filename, target_city, tolerance=2.0):
    """
    Find cities with similar average temperatures to the target city.
    
    Parameters:
    filename (str): Path to the CSV file
    target_city (str): Reference city name
    tolerance (float): Temperature difference threshold in °C
    
    Returns:
    dict: {
        'target_city': str,
        'target_avg_temp': float,
        'similar_cities': [
            {'city': str, 'country': str, 'avg_temp': float, 'difference': float}
        ],
        'tolerance': float
    }
    
    """
    city_temps={} #{city:[temp1,temp2...]}
    countries={}#{city:[country]}
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        temp_count=0
        for row in reader:
            city = row["City"]
            countries[city]=row["Country"]
            temp_str = row["AverageTemperature"]
            if temp_str and temp_str.strip():
                try:
                    temp = float(temp_str)
                except ValueError:
                    continue
                city_temps.setdefault(city,[]).append(temp) #add city to city_temps if no previous data has been added. Append new temperatures to the city data
        if target_city not in city_temps:
            return None
        avg_temp_target_city=sum(city_temps[target_city])/len(city_temps[target_city]) #average temperature of target city
        similar_cities=[]
        for city,temp in city_temps.items():
            if city!=target_city:
                avg_temp=sum(city_temps[city])/len(city_temps[city]) #calculate average temperature of each city
            difference=abs(avg_temp-avg_temp_target_city)
            if difference<=tolerance: #check if similar city temperature
                similar_cities.append({'city': city, 'country': countries[city], 'avg_temp': avg_temp, 'difference': difference}) #append the city's data to the list
    return {
        'target_city': target_city,
        'target_avg_temp': avg_temp_target_city,
        'similar_cities': similar_cities,
        'tolerance': tolerance
    }           

def get_temperature_trends(filename, city_name, window_size=5):
    """
    Calculate temperature trends using moving averages and identify patterns.
    Parameters:
    filename (str): Path to the CSV file
    city_name (str): Name of the city
    window_size (int): Number of years for moving average calculation
    
    Returns:
    dict: {
        'city': str,
        'raw_annual_data': {'YYYY': float},  # Annual averages
        'moving_averages': {'YYYY': float},  # Moving averages
        'trend_analysis': {
            'overall_slope': float,  # °C per year
            'warming_periods': [{'start': year, 'end': year, 'rate': float}],
            'cooling_periods': [{'start': year, 'end': year, 'rate': float}]
        }
    }
    """
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        raw_annual_data={}
        moving_averages={}
        trend_analysis={}
        year_temp_collection={}
        first_year=None 
        last_year=None
        years=[]
        for row in reader:
            if row["City"] == city_name:
                temp_str = row["AverageTemperature"]
                if temp_str and temp_str.strip():
                    try:
                        temp = float(temp_str)
                        year_str = row["dt"][:4]  # YYYY
                        year=int(row["dt"][:4]) #change to int
                        year_temp_collection.setdefault(year,[]).append(temp)
                        if year not in years:
                            years.append(year) #add every year in datset to the list of years
                    except ValueError:
                        continue
                    if first_year==None or year<first_year:
                        first_year=year #initialize starting year
                    if last_year==None or year>last_year:
                        last_year=year #initialize ending year
        warming_periods=[]
        cooling_periods=[]
        for year in years:
            avg_temp=sum(year_temp_collection[year])/len(year_temp_collection[year]) #average temperature
            raw_annual_data[str(year)]=avg_temp 
            #for window size of 5, moving average taken for (Y-2),(Y-1),(Y),(Y+1),(Y+2)
            if year < (first_year+(window_size//2)) or year > (last_year-(window_size//2)):
                moving_averages[str(year)]=None
            else:
                k=window_size//2
                windowvalues=[]#empty list for storing average temperaure of (Y-2),(Y-1),(Y),(Y+1),(Y+2)
                for y in range(year-k,year+k+1):#moving average set range
                    if y in years:
                        windowvalues.append(sum(year_temp_collection[y])/len(year_temp_collection[y])) #find average temperature fo year and append to list
                moving_averages[str(year)]=float(sum(windowvalues)/len(windowvalues)) #moving average temperature for current year in loop
        #overall slope= (end year temp - start year temp)/(end year -start year)
        overall_slope=float((raw_annual_data[str(last_year)]-raw_annual_data[str(first_year)])/(last_year-first_year)) 
        i=0
        while i<len(years)-1:
            start_year=years[i]#initialize period interval start 
            i+=1
            if raw_annual_data[str(years[i])]<raw_annual_data[str(years[i+1])]: 
                while i<len(years)-1 and raw_annual_data[str(years[i])]<raw_annual_data[str(years[i+1])]:#a sub-loop from start year for finding warming/cooling periods
                    i+=1
                rate=float((raw_annual_data[str(years[i])]-raw_annual_data[str(start_year)])/(years[i]-start_year))#find rate of change in temp for the sub-loop
                warming_periods.append({"start": start_year, "end": years[i], "rate": rate})
            elif raw_annual_data[str(years[i])]>raw_annual_data[str(years[i+1])]:
                while i<len(years)-1 and raw_annual_data[str(years[i])]>raw_annual_data[str(years[i+1])]:
                    i+=1
                rate=float((raw_annual_data[str(years[i])]-raw_annual_data[str(start_year)])/(years[i]-start_year))#find rate of change in temp for the sub-loop
                cooling_periods.append({"start": start_year, "end": years[i], "rate": rate})
                
    return {
        'city': city_name,
        'raw_annual_data': raw_annual_data,  # Annual averages
        'moving_averages': moving_averages,  # Moving averages
        'trend_analysis': {
            'overall_slope': overall_slope,  # °C per year
            'warming_periods': warming_periods,
            'cooling_periods': cooling_periods
        }
    }
    # TODO: Your implementation here
    pass


# =============================================================================
# TESTING CODE 
# =============================================================================

def test_api_functions():
    """
    Test all API functions with sample data.
    """
    filename = 'GlobalLandTemperaturesByMajorCity.csv'
    test_city = 'Madras'
    
    print("Testing Temperature Data API")
    print("=" * 40)
    
    # Test basic function
    temps = get_city_temperatures(filename, test_city)
    print(f"Basic function: Found {len(temps)} temperature records")
    
    # Test extremes
    extremes = find_temperature_extremes(filename, test_city)
    print(f"Extremes: Hottest = {extremes['hottest']['temperature']}°C")
    
    # Test seasonal averages
    summer_avg = get_seasonal_averages(filename, test_city, 'summer')
    print(f"Seasonal: Summer average = {summer_avg['average_temperature']:.1f}°C")
    
    # Test decade comparison
    comparison = compare_decades(filename, test_city, 1990, 2000)
    print(f"Decades: Temperature change = {comparison['difference']:.2f}°C")
    
    # Test similar cities
    similar = find_similar_cities(filename, test_city, tolerance=3.0)
    print(f"Similar cities: Found {len(similar['similar_cities'])} matches")
    
    # Test trends
    trends = get_temperature_trends(filename, test_city)
    print(f"Trends: Overall slope = {trends['trend_analysis']['overall_slope']:.4f}°C/year")


if __name__ == "__main__":
    test_api_functions()
