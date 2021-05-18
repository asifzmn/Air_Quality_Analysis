import os
import shutil
import time
from datetime import date, timedelta, datetime
from distutils.dir_util import copy_tree
from os import listdir
from pathlib import Path
from timeit import default_timer as timer
import pandas as pd
from selenium.webdriver import FirefoxProfile, Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

from data_preparation import *


def Scrap(savePath):
    metaFrame = load_metadata()

    era5list = ['Temperature [2 m]', 'Growing degree days [2 m]', 'Temperature [900 hPa]', 'Temperature [850 hPa]',
                'Temperature [800 hPa]', 'Temperature [700 hPa]', 'Temperature [500 hPa]', 'Precipitation amount',
                'Precipitation runoff', 'Relative humidity [2 m]', 'Wind gusts [10 m]', 'Wind speed [10 m]',
                'Wind speed [100 m]', 'Wind speed and direction [900 hPa]', 'Wind speed and direction [850 hPa]',
                'Wind speed and direction [800 hPa]', 'Wind speed and direction [700 hPa]',
                'Wind speed and direction [500 hPa]', 'Wind speed and direction [250 hPa]', 'Total cloud cover',
                'Low, mid and high cloud cover', 'Sunshine duration (minutes)', 'Solar radiation', 'Longwave radiation',
                'UV radiation', 'Pressure [mean sea level]', 'Evapotranspiration']

    gecko_path = "/home/asif/Work/Firefox Web Driver/geckodriver.exe"
    # gecko_path = "/media/az/Study/Work/Firefox Web Driver/geckodriver.exe"

    profile = FirefoxProfile()
    profile.set_preference('browser.download.folderList', 2)
    profile.set_preference('browser.download.manager.showWhenStarting', False)
    # profile.set_preference('browser.download.dir', os.getcwd())
    profile.set_preference('browser.download.dir', savePath)
    # profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'text/plain')
    profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'text/csv')
    # profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'application/octet-stream')

    # profile.set_preference("browser.helperApps.alwaysAsk.force", False)
    # profile.set_preference("browser.download.manager.showWhenStarting",False)

    profile.set_preference('general.warnOnAboutConfig', False)
    profile.update_preferences()
    driver = Firefox(firefox_profile=profile, executable_path=gecko_path)
    driver.get("https://www.meteoblue.com/en/weather/archive/era5/shahbag_bangladesh_7697915")

    driver.find_element_by_id("gdpr_form").click()

    start = timer()
    for index, row in metaFrame.iterrows():
        print(row)
        location = ' '.join(map(str, row[['Latitude', 'Longitude']].values))
        driver.find_element_by_id("gls").send_keys(location + Keys.RETURN)

        searchTable = WebDriverWait(driver, 30).until(expected_conditions.presence_of_all_elements_located(
            (By.XPATH, "//table[@class = 'search_results']//tr")))
        searchTable[1].find_elements_by_xpath(".//td")[1].click()

        factorInput = '/html/body/div[3]/div/main/div/div[2]/form/div[5]/div[1]/span[1]/span[1]/span/ul/li/input'
        factors = WebDriverWait(driver, 30).until(expected_conditions.presence_of_all_elements_located
                                                  ((By.XPATH, factorInput)))[0]
        # factors.send_keys('Total cloud cover' + '\n' )
        factors.send_keys('\n'.join(map(str, era5list)) + '\n')
        time.sleep(1)

        driver.find_element_by_name("submit_csv").click()
        time.sleep(3)
        print(timer() - start)

    time.sleep(30)
    for file, zone in zip(sorted(Path(savePath).iterdir(), key=os.path.getmtime), metaFrame.index.values):
        shutil.move(file, os.path.join(savePath, zone + '.csv'))


def readFile(path, index='Dhaka'):
    meteoInfo = pd.read_csv(os.path.join(path, index + '.csv'), sep=',', skiprows=9)
    meteoInfo = meteoInfo.set_index(pd.to_datetime(meteoInfo['timestamp']))
    meteoInfo.drop(['timestamp'], axis=1, inplace=True)
    meteoInfo = meteoInfo.apply(pd.to_numeric)
    print(meteoInfo.index.values[[0, -1]])
    return meteoInfo


if __name__ == '__main__':
    Scrap(meteoblue_data_path_2019)
    exit()
