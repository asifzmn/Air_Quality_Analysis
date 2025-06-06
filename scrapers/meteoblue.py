import os
import shutil
import time
from pathlib import Path
from datetime import date, timedelta, datetime
from distutils.dir_util import copy_tree
from timeit import default_timer as timer
from selenium.webdriver import FirefoxProfile, Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from time import sleep

from data_preparation import *


def selcet_meteoblue_zones(meta_data):
    india_bd_trajectory_regions = ['NCT', 'West Bengal', 'Uttar Pradesh', 'Haryana', 'Bihar'][:2]
    meta_data = meta_data[(meta_data.Country == 'Bangladesh') | (
            (meta_data.Country == 'India') & (meta_data.Division.isin(india_bd_trajectory_regions)))]
    return meta_data


def prepare_firefox_driver(savePath):
    gecko_path = "/home/asif/Work/Firefox Web Driver/geckodriver.exe"
    # gecko_path = "/media/az/Study/Work/Firefox Web Driver/geckodriver.exe"

    profile = FirefoxProfile()
    profile.set_preference('browser.download.folderList', 2)
    profile.set_preference('browser.download.manager.showWhenStarting', False)
    # profile.set_preference('browser.download.dir', os.getcwd())
    profile.set_preference('browser.download.dir', savePath)
    profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'text/csv')
    # profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'text/plain')
    # profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'application/octet-stream')

    # profile.set_preference("browser.helperApps.alwaysAsk.force", False)
    # profile.set_preference("browser.download.manager.showWhenStarting",False)

    profile.set_preference('general.warnOnAboutConfig', False)
    profile.update_preferences()
    return Firefox(firefox_profile=profile, executable_path=gecko_path)


def Scrap_2019(savePath):
    meta_data = get_metadata()
    meta_data = selcet_meteoblue_zones(meta_data)
    print(meta_data[meta_data.Division == 'NCT'])
    exit()

    era5list = ['Temperature [2 m]', 'Growing degree days [2 m]', 'Temperature [900 hPa]', 'Temperature [850 hPa]',
                'Temperature [800 hPa]', 'Temperature [700 hPa]', 'Temperature [500 hPa]', 'Precipitation amount',
                'Precipitation runoff', 'Relative humidity [2 m]', 'Wind gusts [10 m]', 'Wind speed [10 m]',
                'Wind speed [100 m]', 'Wind speed and direction [900 hPa]', 'Wind speed and direction [850 hPa]',
                'Wind speed and direction [800 hPa]', 'Wind speed and direction [700 hPa]',
                'Wind speed and direction [500 hPa]', 'Wind speed and direction [250 hPa]', 'Total cloud cover',
                'Low, mid and high cloud cover', 'Sunshine duration (minutes)', 'Solar radiation', 'Longwave radiation',
                'UV radiation', 'Pressure [mean sea level]', 'Evapotranspiration']

    driver = prepare_firefox_driver(savePath)

    driver.get("https://www.meteoblue.com/en/weather/archive/era5/shahbag_bangladesh_7697915")

    driver.find_element_by_id("gdpr_form").click()

    start = timer()
    for index, row in meta_data.iterrows():
        print(index)
        location = ' '.join(map(str, row[['Latitude', 'Longitude']].values))
        driver.find_element_by_id("gls").send_keys(location + Keys.RETURN)
        # driver.implicitly_wait(1)
        # driver.find_element_by_id("gls").send_keys(Keys.ENTER)

        searchTable = WebDriverWait(driver, 30).until(expected_conditions.presence_of_all_elements_located(
            (By.XPATH, "//table[@class = 'search-results']//tr")))
        searchTable[1].find_elements_by_xpath(".//td")[1].click()

        # factorInput = '/html/body/div[3]/div/main/div/div[2]/form/div[5]/div[1]/span[1]/span[1]/span/ul/li/input'
        # factors = WebDriverWait(driver, 30).until(expected_conditions.presence_of_all_elements_located
        #                                           ((By.XPATH, factorInput)))[0]
        # # factors.send_keys('Total cloud cover' + '\n' )
        # factors.send_keys('\n'.join(map(str, era5list)) + '\n')

        # WebDriverWait(driver, 30).until(
        #     expected_conditions.element_to_be_clickable((By.CLASS_NAME, 'select-all'))).click()
        select_all_button = driver.find_element_by_class_name('select-all')
        # time.sleep(3)
        # select_all_button.click()
        driver.execute_script("arguments[0].click();", select_all_button)

        driver.find_element_by_name("submit_csv").click()
        time.sleep(5)
        print(timer() - start)

    time.sleep(30)
    for file, zone in zip(sorted(Path(savePath).iterdir(), key=os.path.getmtime), meta_data.Zone.values):
        shutil.move(file, os.path.join(savePath, zone + '.csv'))


def Scrap(savePath):
    list = ['Temperature [1000 hPa]', 'Temperature [850 hPa]', 'Temperature [700 hPa]', 'Wind speed [80 m]',
            'Wind gusts [10 m]', 'Wind speed and direction [900 hPa]', 'Wind speed and direction [850 hPa]',
            'Wind speed and direction [700 hPa]', 'Wind speed and direction [500 hPa]',
            'Sunshine duration (minutes)', 'Solar radiation', 'Direct radiation', 'Diffuse radiation',
            'Precipitation amount', 'Low, mid and high cloud cover', 'Pressure [mean sea level]',
            'Surface skin temperature', 'Soil temperature [0-10 cm down]', 'Soil moisture [0-10 cm down]']

    lastDate, oneDay = datetime.strptime(max(listdir(savePath)).split(' to ')[1], '%Y-%m-%d').date(), timedelta(days=1)
    datePoints = str(lastDate + oneDay) + ' to ' + str(date.today() - oneDay)
    datePointsupdate = str(lastDate + oneDay) + ' - ' + str(date.today() - oneDay)
    targetPath = savePath + datePoints

    driver = prepare_firefox_driver(savePath)
    driver.get("https://www.meteoblue.com/en/weather/archive/export/shahbag_bangladesh_7697915")

    driver.find_element_by_id("gdpr_form").click()

    if not os.path.exists(targetPath): os.makedirs(targetPath)
    start = timer()
    for index, row in metaFrame.iterrows():
        print(row)
        location = ' '.join(map(str, row[['Latitude', 'Longitude']].values))
        driver.find_element_by_id("gls").send_keys(location + Keys.RETURN)

        searchTable = WebDriverWait(driver, 30).until(expected_conditions.presence_of_all_elements_located(
            (By.XPATH, "//table[@class = 'search_results']//tr")))
        searchTable[1].find_elements_by_xpath(".//td")[1].click()

        factorInput = '/html/body/div[3]/div/main/div/div[2]/form/div[5]/div[1]/span[1]/span[1]/span/ul/li/input'
        # factorInput = '//*[@id="wrapper-main"]/div/main/div/div[2]/form/div[5]/div[1]/span/span[1]/span/ul/li[
        # 4]/input'
        factors = WebDriverWait(driver, 30).until(expected_conditions.presence_of_all_elements_located
                                                  ((By.XPATH, factorInput)))[0]
        # factors.send_keys('Total cloud cover' + '\n' )
        factors.send_keys('\n'.join(map(str, list)) + '\n')
        sleep(1)

        # for tickBox in ["relhum2m", "pressure", "clouds", "sunshine", "swrad", "directrad", "diffuserad", "windgust",
        #                 "wind+dir80m", "wind+dir900mb", "wind+dir850mb", "wind+dir700mb", "wind+dir500mb"
        #                 , "temp1000mb","temp850mb", "temp700mb", "tempsfc", "soiltemp0to10", "soilmoist0to10"
        #                 ]: driver.find_element_by_id(tickBox).click()

        datePicker = driver.find_element_by_id('daterange')
        datePicker.clear()
        datePicker.send_keys(datePointsupdate + Keys.RETURN)

        driver.find_element_by_name("submit_csv").click()
        # filename = max([targetPath + "/" + f for f in os.listdir(targetPath)], key=os.path.getctime)
        # shutil.move(filename, os.path.join(targetPath, index + '.csv'))
        time.sleep(3)
        print(timer() - start)

    time.sleep(30)
    for file, zone in zip(sorted(Path(targetPath).iterdir(), key=os.path.getmtime), metaFrame.index.values):
        shutil.move(file, os.path.join(targetPath, zone + '.xlsx'))


def readFile(path, index='Dhaka'):
    meteoInfo = pd.read_csv(os.path.join(path, index + '.csv'), sep=',', skiprows=9)
    meteoInfo = meteoInfo.set_index(pd.to_datetime(meteoInfo['timestamp']))
    meteoInfo.drop(['timestamp'], axis=1, inplace=True)
    meteoInfo = meteoInfo.apply(pd.to_numeric)
    print(meteoInfo.index.values[[0, -1]])
    return meteoInfo


def Update(savePath, savePathcp):
    savedata, runningData = readFile(savePath), readFile(runningPath)
    # print((runningData[savedata.index.values[-1]+1:]))
    # print(pd.concat([savedata, runningData], ignore_index=False).drop_duplicates())

    if input() != 'yes': return
    copy_tree(savePath, savePathcp)
    for index, row in metaFrame.iterrows():
        with open(os.path.join(savePath, index + '.csv'), 'a') as save:
            save.write("\n" + '\n'.join(
                map(str, open(os.path.join(runningPath, index + '.csv')).read().split('\n')[10:])))


if __name__ == '__main__':
    metaFrame = get_metadata()
    runningPath = '/media/az/Study/Air Analysis/AirQuality Dataset/MeteoblueJuly'

    # Scrap(meteoblue_data_path)
    Scrap_2019(meteoblue_data_path_2019)

    # df = pd.read_excel('/home/asif/Work/Air Analysis/AQ Dataset/Meteoblue Scrapped Data/
    # 2021-03-08 to 2021-03-22/Azimpur.xlsx',engine='openpyxl',header=9)
    # df = open('/home/asif/Work/Air Analysis/AQ Dataset/Meteoblue Scrapped Data/
    # 2021-03-08 to 2021-03-22/Azimpur.xlsx','rb').read().decode('unicode_escape')
    # print(df)
    exit()
