from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

from paths import wunderground_data_path
from scrapers import prepare_firefox_driver

if __name__ == '__main__':
    regions = ['Dhaka', 'West Bengal', 'NCT', 'Uttar Pradesh', 'Telangana', 'Shan'][1] + '/'
    region_url_key = ['bd/dhaka/VGHS', 'in/dum-dum/VECC', 'in/new-delhi/VIDP',
                      'in/lucknow/VILK', 'in/hyderabad/VOHS', 'mm/yangon/VYYY'][1]

    # regions = ['San_Francisco', 'Austin', 'Seattle',
    #            'Phoenix', 'Denver', 'Salt_Lake_City',
    #            'Portland', 'Las_Vegas', 'Boise',
    #            'Albuquerque', 'Billings', 'Cheyenne'][6] + '/'
    # region_url_key = ['ca/san-francisco/KSFO', 'tx/austin/KAUS', 'wa/seattle/KSEA',
    #                   'az/phoenix/KPHX', 'co/denver/KDEN', 'ut/salt-lake-city/KSLC',
    #                   'or/portland/KPDX', 'nv/las-vegas/KVGT', 'id/boise/KBOI',
    #                   'nm/albuquerque/KABQ', 'mt/billings/KBIL', 'wy/cheyenne/KCYS'][6]   # usa

    viewButton = '//*[@id="inner-content"]/div[2]/div[1]/div[1]/div[1]/div/lib-date-selector/div/input'
    tableElem = '//*[@id="inner-content"]/div[2]/div[1]/div[5]/div[1]/div/lib-city-history-observation/div/div[2]/table'

    temp_columns = ['Temperature', 'Dew Point']
    string_columns = ['Time', 'Wind', 'Condition']

    # options = webdriver.ChromeOptions()
    # options.add_argument("--start-maximized")
    # prefs = {"translate_whitelists": {"bn": "en"}, "translate": {"enabled": "true"}}
    # options.add_experimental_option("prefs", prefs)
    # driver = webdriver.Chrome('/home/az/.wdm/drivers/chromedriver/linux64/86.0.4240.22/chromedriver', options=options)
    driver = prepare_firefox_driver()

    # timeRange = pd.date_range('2012-01-01', '2016-12-31')  # usa
    timeRange = pd.date_range('2012-05-13', '2016-12-31')

    for single_date in timeRange:
        print(single_date)
        url = f'https://www.wunderground.com/history/daily/{region_url_key}/date/' + str(single_date.date())
        driver.get(url)
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, viewButton))).click()

        table = WebDriverWait(driver, 180).until(EC.presence_of_element_located((By.XPATH, tableElem)))
        headers = [header.text for header in table.find_elements_by_xpath(".//tr")[0].find_elements_by_xpath(".//th")]
        bodyInfo = [[cell.text for cell in row.find_elements_by_xpath(".//td")] for row in
                    table.find_elements_by_xpath(".//tr")[1:]]

        df = pd.DataFrame(data=bodyInfo, columns=headers).set_index('Time')
        df.index = pd.to_datetime(str(single_date.date()) + ' ' + df.index)
        df[df.columns.difference(string_columns)] = df[df.columns.difference(string_columns)].apply(
            lambda x: x.str.split().str[0].astype('float'))
        df[temp_columns] = df[temp_columns].apply(lambda x: ((x - 32) * 5 / 9).round(2))
        # df.agg({'Time': lambda x: pd.to_datetime(str(singleDate.date()) + ' ' + x)})
        save_path = wunderground_data_path + regions + str(single_date.date())
        # save_path = wunderground_data_path + 'USA/' + regions + str(single_date.date())  # usa
        df.to_csv(save_path)

    # headers = ['Time', 'Temperature', 'Dew Point', 'Humidity', 'Wind', 'Wind Speed', 'Wind Gust', 'Pressure', 'Precip.',
    #  'Condition']
    # bodyInfo =[['12:00 AM', '68 F', '61 F', '78 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['12:30 AM', '68 F', '61 F', '78 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['1:00 AM', '68 F', '63 F', '83 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['1:30 AM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['2:00 AM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['2:30 AM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['3:00 AM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['3:30 AM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['4:00 AM', '64 F', '61 F', '88 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['4:30 AM', '64 F', '61 F', '88 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['5:00 AM', '64 F', '61 F', '88 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['5:30 AM', '64 F', '61 F', '88 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['6:00 AM', '63 F', '61 F', '94 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['6:30 AM', '63 F', '61 F', '94 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['7:00 AM', '61 F', '59 F', '94 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['7:30 AM', '61 F', '59 F', '94 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['8:00 AM', '63 F', '61 F', '94 %', 'CALM', '0 mph', '0 mph', '29.97 in', '0.0 in', 'Fog'],
    #  ['8:30 AM', '63 F', '61 F', '94 %', 'CALM', '0 mph', '0 mph', '30.00 in', '0.0 in', 'Fog'],
    #  ['9:00 AM', '63 F', '59 F', '88 %', 'CALM', '0 mph', '0 mph', '30.00 in', '0.0 in', 'Fog'],
    #  ['9:30 AM', '64 F', '61 F', '88 %', 'CALM', '0 mph', '0 mph', '30.03 in', '0.0 in', 'Fog'],
    #  ['10:00 AM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '30.03 in', '0.0 in', 'Fog'],
    #  ['10:30 AM', '68 F', '61 F', '78 %', 'CALM', '0 mph', '0 mph', '30.03 in', '0.0 in', 'Fog'],
    #  ['11:00 AM', '70 F', '61 F', '73 %', 'ENE', '6 mph', '0 mph', '30.00 in', '0.0 in', 'Fog'],
    #  ['11:30 AM', '72 F', '61 F', '69 %', 'CALM', '0 mph', '0 mph', '30.00 in', '0.0 in', 'Haze'],
    #  ['12:00 PM', '73 F', '61 F', '65 %', 'CALM', '0 mph', '0 mph', '29.97 in', '0.0 in', 'Haze'],
    #  ['12:30 PM', '75 F', '61 F', '61 %', 'CALM', '0 mph', '0 mph', '29.97 in', '0.0 in', 'Haze'],
    #  ['1:00 PM', '77 F', '57 F', '50 %', 'NE', '6 mph', '0 mph', '29.94 in', '0.0 in', 'Haze'],
    #  ['1:30 PM', '77 F', '54 F', '44 %', 'N', '6 mph', '0 mph', '29.91 in', '0.0 in', 'Haze'],
    #  ['2:00 PM', '79 F', '54 F', '42 %', 'NNE', '7 mph', '0 mph', '29.91 in', '0.0 in', 'Haze'],
    #  ['2:30 PM', '79 F', '54 F', '42 %', 'NNW', '6 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['3:00 PM', '79 F', '54 F', '42 %', 'NNE', '5 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['3:30 PM', '79 F', '54 F', '42 %', 'N', '6 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['4:00 PM', '77 F', '54 F', '44 %', 'NNW', '6 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['4:30 PM', '77 F', '54 F', '44 %', 'NNW', '6 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['5:00 PM', '75 F', '54 F', '47 %', 'CALM', '0 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['5:30 PM', '75 F', '54 F', '47 %', 'CALM', '0 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['6:00 PM', '73 F', '54 F', '50 %', 'CALM', '0 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['6:30 PM', '73 F', '55 F', '53 %', 'CALM', '0 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['7:00 PM', '72 F', '55 F', '57 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Haze'],
    #  ['7:30 PM', '72 F', '55 F', '57 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Haze'],
    #  ['8:00 PM', '72 F', '55 F', '57 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Haze'],
    #  ['8:30 PM', '70 F', '59 F', '68 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Haze'],
    #  ['9:00 PM', '68 F', '61 F', '78 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['9:30 PM', '68 F', '63 F', '83 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['10:00 PM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['10:30 PM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['11:00 PM', '64 F', '59 F', '83 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['11:30 PM', '64 F', '59 F', '83 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog']]
