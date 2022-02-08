from selenium import webdriver
from selenium.webdriver import FirefoxProfile,Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import statsmodels.api as sm
from AQ_Analysis import *
from sklearn.preprocessing import StandardScaler
import plotly.express as px


def prepare_chrome_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.headless = True
    prefs = {"translate_whitelists": {"bn": "en"}, "translate": {"enabled": "true"}}
    options.add_experimental_option("prefs", prefs)
    # driver = webdriver.Chrome('/home/az/.wdm/drivers/chromedriver/linux64/86.0.4240.22/chromedriver', options=options)
    return webdriver.Chrome('/home/az/Desktop/chromedriver', options=options)


def prepare_firefox_driver():
    gecko_path = "/home/asif/Work/Firefox Web Driver/geckodriver.exe"
    # gecko_path = "/media/az/Study/Work/Firefox Web Driver/geckodriver.exe"

    profile = FirefoxProfile()

    # profile.set_preference("browser.helperApps.alwaysAsk.force", False)
    # profile.set_preference("browser.download.manager.showWhenStarting",False)

    profile.set_preference('general.warnOnAboutConfig', False)
    profile.update_preferences()
    return Firefox(firefox_profile=profile, executable_path=gecko_path)

if __name__ == '__main__':
    FactorAnalysis()
    # VectorAnalysis()
    exit()

    viewButton = '//*[@id="inner-content"]/div[2]/div[1]/div[1]/div[1]/div/lib-date-selector/div/input'
    tableElem = '//*[@id="inner-content"]/div[2]/div[1]/div[5]/div[1]/div/lib-city-history-observation/div/div[2]/table'

    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    prefs = {"translate_whitelists": {"bn": "en"}, "translate": {"enabled": "true"}}
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome('/home/az/.wdm/drivers/chromedriver/linux64/86.0.4240.22/chromedriver', options=options)
    timeRange = pd.date_range('2020-01-01', '2020-12-31')

    for singleDate in timeRange:
        print(singleDate)
        driver.get('https://www.wunderground.com/history/daily/bd/dhaka/VGHS/date/' + str(singleDate.date()))
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, viewButton))).click()

        table = WebDriverWait(driver, 180).until(EC.presence_of_element_located((By.XPATH, tableElem)))
        headers = [header.text for header in table.find_elements_by_xpath(".//tr")[0].find_elements_by_xpath(".//th")]
        bodyInfo = [[cell.text for cell in row.find_elements_by_xpath(".//td")] for row in
                    table.find_elements_by_xpath(".//tr")[1:]]

        df = pd.DataFrame(data=bodyInfo, columns=headers).set_index('Time')
        df.index = pd.to_datetime(str(singleDate.date()) + ' ' + df.index)
        df[df.columns.difference(['Time', 'Wind', 'Condition'])] = df[
            df.columns.difference(['Time', 'Wind', 'Condition'])].apply(lambda x: x.str.split().str[0].astype('float'))
        df[['Temperature', 'Dew Point']] = df[['Temperature', 'Dew Point']].apply(lambda x: ((x - 32) * 5 / 9).round(2))
        # df.agg({'Time': lambda x: pd.to_datetime(str(singleDate.date()) + ' ' + x)})
        df.to_csv('/home/az/PycharmProjects/Data Science/Misc/Past Weather/' + str(singleDate.date()))

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
