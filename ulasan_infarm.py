from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from selenium.webdriver.common.by import By


def scrape_infarm(eksisting_data):
    print("🚀 Memulai browser...")

    url ="https://www.tokopedia.com/kayuan/review"
    # Tunggu halaman dimuat
    print("⏳ Menunggu halaman dimuat...")
    time.sleep(5)

    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    data =[]
    for i in range(0, 3):
        soup = BeautifulSoup(driver.page_source, "html.parser")
        containers = soup.findAll('article', attrs = {'class' : 'css-1pr2lii'})
        for container in containers:
            try:
                nama = container.find('span', attrs = {'data-testid': 'lblItemUlasan'}).text
                produk = container.find('p', attrs = {'class': 'css-d2yr2-unf-heading e1qvo2ff8'}).text
                company = "Infarm"
                data.append([produk,nama, company])
            except AttributeError:
                continue
        time.sleep(2)
        driver.find_element(By.CSS_SELECTOR,"button[aria-label^='Laman berikutnya']").click()
        time.sleep(3)
    # Simpan ke Excel
    print(data)
    print(f"\n✅ Data berhasil disimpan")
    all_data = eksisting_data + data
    return all_data
