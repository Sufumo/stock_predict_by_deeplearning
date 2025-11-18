import requests
import json
import os
from typing import List, Dict, Any
import re

headers = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/bk/90.BK0733.html",
    "Cookie": "qgqp_b_id=90ff9cece2b5376eed839c7647c1a384; st_nvi=l9__Jd2RcKdLnQEfrNoPR3129; nid=0ce3a2e7865907ec479232c109e9c66d; nid_create_time=1760950607621; gvi=LowTRaJdoF1O5Y7zR8K871d95; gvi_create_time=1760950607621; EMFUND1=null; EMFUND2=null; EMFUND3=null; EMFUND0=null; EMFUND4=10-20%2017%3A05%3A11@%23%24%u4E2D%u6B27%u6570%u5B57%u7ECF%u6D4E%u6DF7%u5408%u53D1%u8D77A@%23%24018993; EMFUND5=10-20%2017%3A07%3A41@%23%24%u957F%u57CE%u6536%u76CA%u5B9D%u8D27%u5E01C@%23%24016778; EMFUND6=10-20%2017%3A08%3A34@%23%24%u540C%u6CF0%u4EA7%u4E1A%u5347%u7EA7%u6DF7%u5408A@%23%24014938; EMFUND9=10-20%2017%3A17%3A37@%23%24%u56FD%u5BCC%u7A33%u5065%u517B%u8001%u4E00%u5E74%u6DF7%u5408%28FOF%29Y@%23%24019456; EMFUND7=10-20%2018%3A58%3A58@%23%24%u8DEF%u535A%u8FC8%u8D44%u6E90%u7CBE%u9009%u80A1%u7968%u53D1%u8D77A@%23%24021875; EMFUND8=10-20 18:59:15@#$%u5E73%u5B89%u5148%u8FDB%u5236%u9020%u4E3B%u9898%u80A1%u7968%u53D1%u8D77A@%23%24019457; emshistory=%5B%22%E7%BE%8E%E7%9A%84%E9%9B%86%E5%9B%A2%22%5D; websitepoptg_api_time=1763345092393; fullscreengg=1; fullscreengg2=1; st_si=19369909345516; st_asi=delete; st_pvi=27913905723683; st_sp=2025-11-17%2013%3A13%3A59; st_inirUrl=https%3A%2F%2Fquote.eastmoney.com%2Fbk%2F90.BK0733.html; st_sn=4; st_psi=20251117134137952-113200301201-8392114807"
}

def get_industry_list() -> List[List[str]]:
    """
    获取行业列表信息
    返回格式: [["包装材料", 90, "BK0733"], ...]
    """
    url = "https://push2.eastmoney.com/api/qt/clist/get?np=1&fltt=1&invt=2&cb=jQuery37107908227323297126_1763351050341&fs=m%3A90%2Bt%3A2%2Bf%3A!50&fields=f12%2Cf13%2Cf14%2Cf1%2Cf2%2Cf4%2Cf3%2Cf152%2Cf20%2Cf8%2Cf104%2Cf105%2Cf128%2Cf140%2Cf141%2Cf207%2Cf208%2Cf209%2Cf136%2Cf222&fid=f3&pn=1&pz=100&po=1&dect=1&ut=fa5fd1943c7b386f172d6893dbfba10b&wbp2u=%7C0%7C0%7C0%7Cweb&_=1763351050346"
    
    try:
        with requests.get(url, headers=headers) as response:
            resText = response.text
        
        # 使用正则表达式提取JSON数据
        json_match = re.search(r'jQuery\d+_\d+\((.*)\)', resText)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            
            industry_list = []
            if 'data' in data and 'diff' in data['data']:
                for item in data['data']['diff']:
                    f12 = item.get('f12', '')  # 行业代码
                    f13 = item.get('f13', '')  # 市场代码
                    f14 = item.get('f14', '')  # 行业名称
                    
                    if f12 and f13 and f14:
                        industry_list.append([f14, f13, f12])
            
            return industry_list
        else:
            print("无法解析返回数据")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        return []

def get_industry_kline(secid: str) -> Dict[str, List[List[str]]]:
    """
    获取行业K线数据
    secid格式: "{f13}.{f12}" 如 "90.BK0733"
    返回格式: {"行业名称": [["2025-11-14","1299.76",...], ...]}
    """
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?cb=jQuery35101290006374981434_1763352104194&secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&end=20500101&lmt=100000&_=1763352104233"
    
    try:
        with requests.get(url, headers=headers) as response:
            
            resText = response.text
        
        # 使用正则表达式提取JSON数据
        json_match = re.search(r'jQuery\d+_\d+\((.*)\)', resText)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            
            result = {}
            if 'data' in data:
                industry_name = data['data'].get('name', '')
                klines = data['data'].get('klines', [])
                
                # 解析K线数据
                parsed_klines = []
                for kline in klines:
                    # 按逗号分割每个字段
                    fields = kline.split(',')
                    parsed_klines.append(fields)
                
                result[industry_name] = parsed_klines
                return result
            else:
                return {}
        else:
            print(f"无法解析返回数据 for secid: {secid}")
            return {}
            
    except requests.exceptions.RequestException as e:
        print(f"请求失败 for secid {secid}: {e}")
        return {}
    except json.JSONDecodeError as e:
        print(f"JSON解析失败 for secid {secid}: {e}")
        return {}

def collect_industry_data(output_dir: str):
    """
    整合函数：收集所有行业数据并保存到指定目录
    """
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, "industry_kline_data.json")
    
    # 检查数据文件是否已存在
    if os.path.exists(output_file):
        print(f"数据文件已存在: {output_file}")
        print("跳过数据收集步骤")
        return
    
    print("开始收集行业数据...")
    
    # 第一步：获取行业列表
    print("获取行业列表中...")
    industry_list = get_industry_list()
    
    if not industry_list:
        print("获取行业列表失败")
        return
    
    print(f"共获取到 {len(industry_list)} 个行业")
    
    # 第二步：循环获取每个行业的K线数据
    all_industry_data = {}
    success_count = 0
    
    for i, industry in enumerate(industry_list, 1):
        industry_name, market_code, industry_code = industry
        secid = f"{market_code}.{industry_code}"
        
        print(f"正在获取 {industry_name} 的数据 ({i}/{len(industry_list)})...")
        
        kline_data = get_industry_kline(secid)
        
        if kline_data:
            all_industry_data.update(kline_data)
            success_count += 1
            print(f"  ✓ 成功获取 {industry_name} 数据")
        else:
            print(f"  ✗ 获取 {industry_name} 数据失败")
        
        # 添加延迟避免请求过快
        import time
        time.sleep(3)
    
    # 第三步：保存数据到文件
    if all_industry_data:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_industry_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n数据收集完成！")
            print(f"成功获取 {success_count}/{len(industry_list)} 个行业的数据")
            print(f"数据已保存到: {output_file}")
            
            # 显示数据统计信息
            total_kline_records = sum(len(klines) for klines in all_industry_data.values())
            print(f"总K线记录数: {total_kline_records}")
            
        except Exception as e:
            print(f"保存数据失败: {e}")
    else:
        print("未能获取到任何行业数据")

# 使用示例
if __name__ == "__main__":
    # 设置输出目录
    output_directory = "./industry_data"
    
    print(get_industry_list())
    
    # 执行数据收集
    # collect_industry_data(output_directory)
    
    # 测试单个函数
    # industry_list = get_industry_list()
    # print("行业列表:", industry_list[:5])  # 显示前5个行业
    
    # if industry_list:
    #     test_secid = f"{industry_list[0][1]}.{industry_list[0][2]}"
    #     kline_data = get_industry_kline(test_secid)
    #     print("测试K线数据:", list(kline_data.keys())[0] if kline_data else "无数据")