from selenium import webdriver
from selenium.webdriver import FirefoxProfile, Firefox

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
