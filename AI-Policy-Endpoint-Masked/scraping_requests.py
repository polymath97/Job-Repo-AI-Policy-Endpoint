from selenium import webdriver


from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC


url = 'https://www.imf.org/en/Search#q=AI%20Strategy.&sort=relevancy&numberOfResults=50'
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
options.add_argument('--ignore-certificate-errors')
driver = webdriver.Chrome(options=options)
driver.get(url)



# wait 15 seconds for all elements to be located
elements = WebDriverWait(driver, 15).until(
    EC.visibility_of_all_elements_located((By.CSS_SELECTOR, '.imf-result-item .CoveoResultLink'))
)

# Extract titles and URLs from elements
for element in elements:
    title = element.get_attribute('title')
    url = element.get_attribute('href')
    print("Title:", title)
    print("URL:", url)

driver.quit()