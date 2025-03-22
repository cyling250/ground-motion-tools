import unittest

if __name__ == "__main__":
    # dicsover方法查找用例
    suite = unittest.defaultTestLoader.discover("", "*test.py")
    # 打开文件对象
    with open("test_report.txt", "w") as f:
        # TextTestRunner运行用例
        runer = unittest.TextTestRunner(stream=f, verbosity=2)  # verbosity=2 输出详细日志
        runer.run(suite)
