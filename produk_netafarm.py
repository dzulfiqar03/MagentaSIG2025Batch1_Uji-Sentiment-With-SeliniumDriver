from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from selenium.webdriver.common.by import By
# from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException

def scrape_prodnetafarm():
    print("🚀 Memulai browser...")

    url ="https://www.tokopedia.com/netafarm/product"
    # Tunggu halaman dimuat
    print("⏳ Menunggu halaman dimuat...")
    time.sleep(5)


    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    dataProduk =[]
    for i in range(0, 10):
        soup = BeautifulSoup(driver.page_source, "html.parser")
        containers = soup.findAll('a', attrs = {'class' : 'oQ94Awb6LlTiGByQZo8Lyw== IM26HEnTb-krJayD-R0OHw=='})
        for container in containers:
            try:
                nama = container.find('span').text
                harga = container.find('div', attrs = {'class' : '_67d6E1xDKIzw+i2D2L0tjw=='}).text
                dataProduk.append([nama,harga])
            except AttributeError:
                continue
        time.sleep(2)
        driver.find_element(By.CSS_SELECTOR,"a[data-testid^='btnShopProductPageNext']").click()
        time.sleep(3)
        
    # while True:
    #     soup = BeautifulSoup(driver.page_source, "html.parser")
    #     containers = soup.findAll('a', attrs={'class': 'oQ94Awb6LlTiGByQZo8Lyw== IM26HEnTb-krJayD-R0OHw=='})

    #     for container in containers:
    #         try:
    #             nama = container.find('span').text
    #             harga = container.find('div', attrs={'class': '_67d6E1xDKIzw+i2D2L0tjw=='}).text
    #             dataProduk.append([nama, harga])
    #         except AttributeError:
    #             continue

    #     time.sleep(2)

    #     try:
    #         next_button = driver.find_element(By.CSS_SELECTOR, "a[data-testid^='btnShopProductPageNext']")
    #         next_button.click()
    #         time.sleep(3)
    #     except (NoSuchElementException, ElementClickInterceptedException):
    #         print("🚫 Tidak ada tombol 'Next' lagi. Proses scraping selesai.")
    #         break
    
    # Simpan ke Excel
    print(dataProduk)
   
    print(f"\n✅ DataProduk berhasil disimpan")
    return dataProduk
