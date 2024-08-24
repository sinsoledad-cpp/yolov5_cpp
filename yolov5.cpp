#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <string>
#include <vector>
/*
    前处理
    推理
    后处理：筛选置信度过低的目标，按类别分类并得到类别标号，筛选重复度过高的框，结束

    筛选重复度过高的框策略1：拿到置信度最高的框，把剩下的去掉
    筛选重复度过高的框策略2：拿到置信度最高的框，看剩下的框是否重复，重复过多就删除，也就是所谓的nms
*/
static const std::vector<std::string> class_name = { "animals","cat","chicken", "cow","dog","fox","goat","horse","person","racoon", "skunk" };

void print_result(const cv::Mat& result,float conf=0.7,int len_data=16)
{
    std::cout << result.total() << std::endl;
    float* pdata = (float*)result.data;
    for (int i = 0; i < result.total() / len_data; i++)
    {
        if(pdata[4]>conf)
        {
            for (int j = 0; j < len_data; j++)
            {
                std::cout << pdata[j] << " ";
            }
            std::cout << std::endl;
        }
        pdata += len_data;
    }
    return;
}
std::vector<std::vector<float>> get_info(const cv::Mat& result, float conf = 0.7, int len_data = 16)
{
    std::cout << result.total() << std::endl;
    float* pdata = (float*)result.data;
    std::vector<std::vector<float>> info;
    for (int i = 0; i < result.total() / len_data; i++)
    {
        if (pdata[4] > conf)
        {
            std::vector<float> info_line;
            for (int j = 0; j < len_data; j++)
            {
                info_line.push_back(pdata[j]);
            }
            info.push_back(info_line);
        }
        pdata += len_data;
    }
    return info;
}

void info_simplify(std::vector<std::vector<float>>& info)
{
    for (auto i = 0; i < info.size(); i++)
    {
        info[i][5] = std::max_element(info[i].cbegin() + 5, info[i].cend()) - (info[i].cbegin() + 5);
        info[i].resize(6);
        float x = info[i][0];
        float y = info[i][1];
        float w = info[i][2];
        float h = info[i][3];
        info[i][0] = x - w / 2.0;
        info[i][1] = y - h / 2.0;
        info[i][2] = x + w / 2.0;
        info[i][3] = y + h / 2.0;
    }
}
std::vector<std::vector<std::vector<float>>> split_info(std::vector<std::vector<float>>& info)
{
    std::vector<std::vector<std::vector<float>>> info_split;
    std::vector<int> class_id;
    for (auto i = 0; i < info.size(); i++)
    {
        if (std::find(class_id.begin(), class_id.end(), (int)info[i][5]) == class_id.end())
        {
            class_id.push_back((int)info[i][5]);
            std::vector<std::vector<float>> info_;
            info_split.push_back(info_);
        }
        info_split[std::find(class_id.begin(), class_id.end(), (int)info[i][5]) - class_id.begin()].push_back(info[i]);
    }
    return info_split;
}

void nms(std::vector<std::vector<float>>& info,float iou=0.4)
{
    int counter = 0;
    std::vector<std::vector<float>> return_info;
    while (counter < info.size())
    {

        return_info.clear();
        float x1 = 0;
        float x2 = 0;
        float y1 = 0;
        float y2 = 0;
        std::sort(info.begin(), info.end(), [](std::vector<float> p1, std::vector<float> p2)
            {
                return p1[4] > p2[4];
            });
        for (auto i = 0; i < info.size(); i++)
        {
            if (i < counter)continue;
            if (i == counter)
            {
                x1 = info[i][0];
                y1 = info[i][1];
                x2 = info[i][2];
                y2 = info[i][3];
                return_info.push_back(info[i]);
                continue;
            }
            if (info[i][0] > x2 || info[i][2]<x1 || info[i][1]>y2 || info[i][3] < y1)
            {
                return_info.push_back(info[i]);
            }
            else
            {
                float over_x1 = std::max(x1, info[i][0]);
                float over_y1 = std::max(y1, info[i][1]);
                float over_x2 = std::max(x2, info[i][2]);
                float over_y2 = std::max(y2, info[i][3]);
                float s_over = (over_x2 - over_x1) * (over_y2 - over_y1);
                float s_total = (x2 - x1) * (y2 - y1) + (info[i][0] - info[i][2]) * (info[i][1] - info[i][3]) - s_over;
                if (s_over / s_total < iou)
                {
                    return_info.push_back(info[i]);
                }
            }
        }
        info = return_info;
        counter += 1;
    }
}

void print_info(const std::vector<std::vector<float>>& info)
{
    for (auto i = 0; i < info.size(); i++)
    {
        for (auto j = 0; j < info[i].size(); j++)
        {
            std::cout << info[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void draw_box(cv::Mat& img, const std::vector<std::vector<float>>& info)
{
    for (auto i = 0; i < info.size(); i++)
    {
        cv::rectangle(img, cv::Point(info[i][0], info[i][1]), cv::Point(info[i][2], info[i][3]), cv::Scalar(0, 255, 0));
        std::string label;
        label += class_name[info[i][5]];
        label += "  ";
        std::stringstream oss;
        oss << info[i][4];
        label += oss.str();
        cv::putText(img, label, cv::Point(info[i][0], info[i][1]), 1, 2, cv::Scalar(0, 255, 0), 2);
    }
}

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);//不再输出日志
    cv::dnn::Net net = cv::dnn::readNetFromONNX("best.onnx");
    cv::Mat img = cv::imread("fox.jpg");
    cv::resize(img, img, cv::Size(640, 640));
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true);
    net.setInput(blob);
    std::vector<cv::Mat> netoutput;
    std::vector<std::string> out_name = {"output"};
    net.forward(netoutput, out_name);
    cv::Mat result = netoutput[0];
    print_result(result);
    std::vector<std::vector<float>> info = get_info(result);
    info_simplify(info);
    print_info(info);
    std::vector<std::vector<std::vector<float>>> info_split = split_info(info);
    print_info(info_split[0]);
    std::cout << info.size() << " " << info[0].size() << std::endl;
    nms(info_split[0]);
    std::cout << "nms" << std::endl;
    print_info(info_split[0]);
    draw_box(img, info_split[0]);
    cv::imshow("test", img);
    cv::waitKey(0);
    return 0;
}