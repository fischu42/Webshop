from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_mediamarkt_links(pages=1):
    options = Options()

    driver = webdriver.Chrome(options=options)
    base_url = "https://www.mediamarkt.hu/hu/search.html?query=telefon&page="

    hrefs = set()

    for page in range(1, pages + 1):
        url = f"{base_url}{page}"
        driver.get(url)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="/hu/product/_"]'))
        )

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        product_links = soup.select('a[href*="/hu/product/_"]')

        for link in product_links:
            href = link.get('href')
            if href:
                hrefs.add("https://www.mediamarkt.hu" + href)

    driver.quit()
    return list(hrefs)

def extract_product_data(url, driver):
    driver.get(url)
    time.sleep(3)  # Allow JS to load

    soup = BeautifulSoup(driver.page_source, "html.parser")
    tables = soup.find_all("table", class_="sc-43bf5cfc-0 gVHLDp")[:8]
    data_dict = {}

    # Extract price
    price_tag = soup.find_all("span", class_="sc-e0c7d9f7-0 bPkjPs")[:2]
    if price_tag:
        data_dict["reviews"] = price_tag[0].get_text(strip=True)
        data_dict["price"] = price_tag[1].get_text(strip=True)
    else:
        data_dict["price"] = None  # or "N/A"
        data_dict["reviews"] = None

    # Extract spec table
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            key_tag = row.find("p", class_="sc-5a9f6c31-0 ejfRGP")
            value_tag = row.find("p", class_="sc-5a9f6c31-0 bMHOgx")
            if key_tag and value_tag:
                key = key_tag.get_text(strip=True)
                value = value_tag.get_text(strip=True)
                data_dict[key] = value

    # Add URL
    data_dict["URL"] = url

    return data_dict

if __name__ == "__main__":
    product_links = scrape_mediamarkt_links(pages=20)
    print(f"Found {len(product_links)} product links.")

    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    all_products = []

    for i, link in enumerate(product_links):
        try:
            print(f"Scraping ({i+1}/{len(product_links)}): {link}")
            product_data = extract_product_data(link, driver)
            all_products.append(product_data)
        except Exception as e:
            print(f"Failed to scrape {link}: {e}")

    driver.quit()

    # Create DataFrame and export
    df = pd.DataFrame(all_products)
    df.to_excel("mediamarkt_products.xlsx", index=False)
    print("Saved to mediamarkt_products.xlsx")
