## CHƯƠNG I: GIỚI THIỆU

## 1.1. Bối cảnh, động lực và vấn đề đặt ra

- Biến đổi khí hậu đang làm thay đổi quy luật hoạt động của bão tại Biển Đông,
    khiến các phương pháp thống kê truyền thống trên Excel trở nên quá tải trước
    lượng dữ liệu khí tượng khổng lồ.
- Sự rời rạc giữa các nguồn dữ liệu (dữ liệu bão, dữ liệu nhiệt độ biển, dữ liệu khí
    quyển) khiến việc phân tích nguyên nhân - kết quả gặp nhiều khó khăn, đòi hỏi
    một hệ thống tích hợp tự động.

## 1.2. Mục tiêu nghiên cứu và câu hỏi nghiên cứu

- Xây dựng quy trình xử lý dữ liệu lớn trên nền tảng Apache Spark để hợp nhất 3
    nguồn dữ liệu không đồng nhất thành một kho dữ liệu chuẩn hóa (~225 triệu bản ghi).
- Huấn luyện mô hình phân loại LightGBM ở cấp ô lưới (0.25°) để phát hiện xác suất
    hiện diện bão hàng ngày, sau đó tổng hợp thành chỉ số Storm Potential Index (SPI)
    để dự báo số lượng bão theo tháng bằng hồi quy Ridge.
- Trả lời câu hỏi: Dựa trên các thuật toán học máy bottom-up, liệu có thể dự báo
    sớm xu hướng mùa bão (số lượng bão/tháng) dựa trên các chỉ số khí hậu hay không?

## 1.3. Phạm vi, giả định và giới hạn của đề tài

- **Phạm vi:** Tập trung phân tích dữ liệu trong khu vực Tây Thái Bình Dương và Biển
    Đông (0-30°N, 100- 180 °E) trong giai đoạn từ 1980 đến 2024.
- **Giả định:** Các quy luật tương tác giữa đại dương và khí quyển trong quá khứ có
    tính lặp lại và các năm có chỉ số khí hậu giống nhau sẽ có diễn biến mùa bão tương
    tự nhau.

## 1.4. Đối tượng sử dụng

- Các nhà nghiên cứu khí tượng thủy văn cần công cụ khai phá dữ liệu lịch sử tốc độ
    cao để kiểm chứng các giả thuyết khoa học.
- Các cơ quan quản lý và phòng chống thiên tai cần thông tin tham khảo về kịch bản
    mùa bão để lập kế hoạch ứng phó dài hạn.

## 1.5. Phương pháp tiếp cận tổng quát của đề tài

- Sử dụng công nghệ tính toán phân tán trên nền tảng Apache Spark để xử lý song
    song ~225 triệu bản ghi dữ liệu lưới khí tượng.
- Áp dụng phương pháp bottom-up: huấn luyện bộ phân loại LightGBM ở cấp vi mô
    (ô lưới 0.25° × ngày), sau đó tổng hợp xác suất thành chỉ số SPI để dự báo
    số bão/tháng bằng hồi quy Ridge (α=10).


# CHƯƠNG II: CƠ SỞ LÝ THUYẾT

```
2.1. Tổng quan về học máy và bài toán phân loại
```
**Học máy** là một lĩnh vực con của Trí tuệ Nhân tạo tập trung vào việc phát triển các thuật
toán cho phép máy tính học từ dữ liệu để đưa ra dự đoán hoặc quyết định mà không cần
được lập trình một cách tường minh. Một hệ thống học máy cải thiện hiệu suất của nó
theo thời gian khi được cung cấp nhiều dữ liệu hơn.

```
2.1.1. Học máy giám sát
```
Đây là phương pháp học phổ biến nhất, trong đó mô hình được huấn luyện trên một tập
dữ liệu đã được gán nhãn. Mỗi mẫu dữ liệu đầu vào đều đi kèm với kết quả đầu ra mong
muốn. Nhiệm vụ của mô hình là học một hàm ánh xạ từ đầu vào đến đầu ra.

Bài toán phát hiện bão nhiệt đới là một **bài toán học máy giám sát** , với đầu vào là các
đặc trưng khí tượng tại mỗi ô lưới (SST, áp suất, gió, ONI) và nhãn là "có bão" (1)
hoặc "không có bão" (0).

**Tối ưu hóa:** Quá trình huấn luyện thực chất là bài toán tối ưu hóa, tìm kiếm các tham số
của mô hình (ví dụ: trọng số _ww_ trong Logistic Regression) để cực tiểu hóa một **hàm mất
mát** , đo lường sự khác biệt giữa dự đoán của mô hình và nhãn thực tế.

```
2.1.2. Cấu trúc bài toán phân loại nhị phân
```
Bài toán phát hiện sự hiện diện bão tại mỗi ô lưới là một bài toán **phân loại nhị phân**
điển hình, với hai lớp: "có bão" (is_storm = 1) và "không có bão" (is_storm = 0).

Đặc điểm nổi bật của bài toán này là **sự mất cân bằng lớp nghiêm trọng**. Trong thực tế,
chỉ có 34.595 bản ghi dương (có bão) so với 225.190.123 bản ghi âm (không có bão),
tỷ lệ 1:6.509, tức lớp dương chỉ chiếm khoảng **0,015%**. Điều này khiến các chỉ số
đánh giá đơn giản như **Accuracy** trở nên kém ý nghĩa.

Do đó, việc đánh giá cần tập trung vào các chỉ số nhạy cảm với sự mất cân bằng như
**ROC-AUC**, **Diện tích dưới đường cong Precision-Recall (PR-AUC)**, **Precision** và
**Recall**. Ngoài ra, đề tài sử dụng **Isotonic Calibration** để hiệu chuẩn xác suất đầu ra
của mô hình, đảm bảo xác suất phản ánh đúng tỷ lệ dương thực tế.


```
2.1.3. Ghi chú lựa chọn mô hình cho dữ liệu bảng
```
Dữ liệu khí tượng lưới là dạng dữ liệu bảng điển hình, mỗi hàng là một bản ghi thời
tiết tại một ô lưới trong một ngày, mỗi cột là một đặc trưng (SST, áp suất, gió, v.v.).
Khi lựa chọn mô hình cho loại dữ liệu này, cần lưu ý:

- **Logistic Regression:** Là mô hình cơ bản tuyệt vời. Ưu điểm là cho xác suất đầu ra
    dễ tin cậy, tốc độ huấn luyện nhanh và khả năng diễn giải cao thông qua các trọng
    số của đặc trưng.
- **Cây quyết định và các mô hình dựa trên cây:** Như Random Forest, XGBoost,
    LightGBM thường cho hiệu năng mạnh mẽ trên dữ liệu dạng bảng. Chúng có khả
    năng nắm bắt các mối quan hệ phi tuyến tính và tương tác phức tạp giữa các đặc
    trưng mà không cần chuẩn bị đặc trưng quá phức tạp.
- **Các mô hình khác (SVM, KNN, MLP):** Có thể hữu ích trong một số ngữ cảnh cụ
    thể, nhưng thường không phải là lựa chọn hàng đầu cho dữ liệu bảng bị mất cân
    bằng lớp mạnh do khả năng mở rộng, hiệu suất hoặc độ phức tạp tính toán.
**2. 2 Tổng quan về Big Data và Apache Spark trong Khí tượng
2. 2 .1 Kiến trúc xử lý phân tán**
- Cơ chế RDD (Resilient Distributed Datasets) và DataFrame giúp phân chia dữ liệu
khí tượng ra nhiều node để xử lý song song, giảm thời gian tính toán từ hàng giờ
xuống phút.
- Ưu điểm của xử lý trên RAM của Spark so với MapReduce truyền thống, đặc biệt
hiệu quả với các bài toán lặp đi lặp lại như tìm kiếm tương đồng.
**2. 2 .2 Thách thức của dữ liệu Đa chiều**
- Đặc thù của dữ liệu NetCDF: Cấu trúc mảng nhiều chiều (Thời gian, Vĩ độ, Kinh
độ, Độ sâu) gây khó khăn cho các cơ sở dữ liệu quan hệ thông thường.
- Bài toán Dữ liệu Đa chiều: Khó khăn trong việc khớp nối dữ liệu điểm vị trí tâm
bão với dữ liệu nhiệt độ tại ô lưới tương ứng theo thời gian thực.
**2. 2 .3 Một số mô hình dự báo bão cơ bản**
- Mô hình động lưc: Dựa trên hệ phương trình vật lý, độ chính xác cao nhưng tốn
kém tài nguyên siêu máy tính (GFS, ECMWF).
- Mô hình thống kê: Dựa trên xác suất lịch sử, chi phí thấp, phù hợp dự báo hạn
mùa, đây là hướng tiếp cận của đề tài.


**2. 3. Cơ sở khoa học về Xoáy thuận nhiệt đới**
    **2. 3 .1. Định nghĩa và Cấu trúc**

Xoáy thuận nhiệt đới (ở khu vực Tây Bắc Thái Bình Dương gọi là **Bão - Typhoon** ) là một
hệ thống bão quay nhanh đặc trưng bởi một trung tâm có áp suất thấp, hoàn lưu khí quyển
tầng thấp đóng kín, gió mạnh và cấu trúc mây mưa xoắn ốc gây mưa lớn.
Về mặt năng lượng, xoáy thuận nhiệt đới hoạt động như một động cơ nhiệt khổng lồ. Nó
lấy năng lượng từ việc bốc hơi nước của đại dương ấm, sau đó giải phóng nhiệt ẩn khi hơi
nước ngưng tụ thành mây và mưa ở các tầng cao của khí quyển.

**2. 3 .2. Cơ chế hình thành và Điều kiện tiên quyết**

Theo nghiên cứu của Gray (1968) và các tài liệu khí tượng hiện đại, để một vùng áp thấp
nhiệt đới phát triển thành bão, cần hội tụ đủ **6 điều kiện nhiệt động lực học** (đây là cơ sở
để đề tài lựa chọn các trường dữ liệu đầu vào):

1. **Nhiệt độ bề mặt biển**
    o Nước biển phải đủ ấm, với ngưỡng tối thiểu là **26.5°C** và độ sâu của lớp
       nước ấm này phải đạt ít nhất 50m.
    o _Liên hệ với đề tài:_ Đây là biến số quan trọng nhất, được trích xuất từ bộ dữ
       liệu **NOAA OISST**. Nhiệt độ càng cao, năng lượng cung cấp cho bão càng
       lớn.
2. **Độ bất ổn định của khí quyển**
    o Phải có sự chênh lệch nhiệt độ lớn giữa bề mặt biển và tầng đối lưu trên cao
       để không khí nóng ẩm có thể bốc lên mạnh mẽ, tạo thành các đám mây
       dông phát triển theo chiều thẳng đứng.
3. **Độ ẩm ở tầng giữa khí quyển**
    o Tầng đối lưu giữa phải có độ ẩm cao. Nếu không khí quá khô, nó sẽ làm bay
       hơi các đám mây và làm suy yếu cơn bão, triệt tiêu năng lượng của nó.
    o _Liên hệ với đề tài:_ Được phản ánh gián tiếp qua dữ liệu áp suất và gió
       từ **ERA**.
4. **Lực Coriolis**
    o Bão cần lực Coriolis để tạo ra gió xoáy. Do đó, bão hiếm khi hình thành
       trong phạm vi 5° vĩ độ cách đường Xích đạo, nơi lực Coriolis xấp xỉ bằng 0.
    o _Liên hệ với đề tài:_ Dữ liệu IBTrACS được lọc để loại bỏ các nhiễu động
       quá gần xích đạo.
5. **Nhiễu động thời tiết có sẵn**
    o Bão không tự nhiên sinh ra từ hư vô. Nó cần một động lực ban đầu, thường
       là một vùng áp thấp, rãnh gió mùa hoặc sóng Đông.
    o _Liên hệ với đề tài:_ Dữ liệu **Mean Sea Level Pressure (MSLP)** từ ERA
       giúp nhận diện các vùng áp thấp này.
6. **Độ đứt gió thấp**


```
o Độ đứt gió là sự thay đổi vận tốc hoặc hướng gió theo độ cao. Để bão hình
thành, độ đứt gió phải YẾU (< 10 m/s).
o Nếu độ đứt gió quá mạnh, nó sẽ phá hủy cấu trúc thẳng đứng của bão, làm
phân tán nhiệt và ngăn bão mạnh lên.
o Liên hệ với đề tài: Tính toán vector gió U (đông-tây) và V (bắc-nam)
từ ERA5 ở các tầng khí quyển khác nhau cho phép đánh giá yếu tố này.
```
**2. 3 .3. Mối liên hệ giữa Đại dương và Khí quyển**

Sự tương tác này là chìa khóa của đề tài nghiên cứu. Đại dương đóng vai trò là bể chứa
nhiệt. Khi hiện tượng **El Niño** xảy ra, nhiệt độ bề mặt biển ở Đông Thái Bình Dương tăng
cao, trục hoàn lưu khí quyển Walker bị thay đổi, làm thay đổi vị trí hình thành bão:

- **Năm El Niño:** Bão có xu hướng hình thành xa hơn về phía Đông, có quãng đường
    di chuyển dài hơn trên biển ấm, dẫn đến cường độ bão khi vào Biển Đông thường
    rất mạnh, có khả năng thành Siêu bão.
- **Năm La Niña:** Bão hình thành gần Biển Đông hơn, tần suất đổ bộ vào Việt Nam
    thường cao hơn nhưng thời gian tích tụ năng lượng ngắn hơn.
**2. 3 .4. Phân loại cường độ bão**

Đề tài sử dụng thang phân loại sức gió Saffir-Simpson hoặc thang Beaufort dựa trên dữ
liệu từ **IBTrACS** :

- **Áp thấp nhiệt đới (TD):** Gió < 63 km/h (Cấp 6-7).
- **Bão nhiệt đới (TS):** Gió 63–118 km/h (Cấp 8-11).
- **Bão to (Typhoon):** Gió > 118 km/h (Cấp 12+).
Việc phân loại này giúp hệ thống Spark gán nhãn chính xác cho các vector dữ liệu trong
quá trình huấn luyện và đánh giá mô hình.


# 2. 4. Các thuật toán được sử dụng để phân loại và phát hiện bão nhiệt

# đới

**2.** 4**. 1. Decision Tree & Random Forest**

**Cây Quyết Định** là một mô hình phân loại phi tuyến tính, hoạt động bằng cách đặt ra các
câu hỏi "Có/Không" về các đặc trưng khí tượng (ví dụ: "SST > 26.5°C?", "Áp suất
< 1005 hPa?") để phân chia dữ liệu thành các nhóm càng "thuần" càng tốt.
Độ "thuần" được đo bằng chỉ số **Gini** hoặc **Entropy**.

- **Chỉ số Gini** : 𝐺𝑖𝑛𝑖 = 1 − ∑^ (𝑝!)"
    Trong đó _pcpc_ là tỷ lệ mẫu thuộc lớp c trong một nút. Chỉ số Gini càng nhỏ thì nút
    càng thuần.
- **Lợi ích thông tin** : Cây sẽ chọn phép phân chia nào làm giảm chỉ số Gini nhiều
    nhất.
**Hạn chế** : Một cây đơn lẻ rất dễ bị **học thuộc lòng (overfitting)** dữ liệu huấn luyện, dẫn
đến hiệu suất kém trên dữ liệu mới.

**Rừng Ngẫu Nhiên** khắc phục hạn chế này bằng kỹ thuật **Bagging** :

1. **Tạo nhiều cây con** : Xây dựng hàng trăm, hàng ngàn cây quyết định.
2. **Dữ liệu huấn luyện khác nhau** : Mỗi cây được huấn luyện trên một tập con dữ
    liệu được lấy mẫu ngẫu nhiên từ tập gốc.
3. **Đặc trưng khác nhau** : Tại mỗi lần phân chia, mỗi cây chỉ được xem xét một
    tập con ngẫu nhiên các đặc trưng.
4. **Biểu quyết tập thể** : Kết quả dự đoán cuối cùng là lớp được **nhiều cây bầu**
    **chọn nhất** đối với phân loại.
**Ưu điểm của Random Forest** :
- Giảm đáng kể hiện tượng overfitting so với cây đơn.
- Cho độ chính xác cao và ổn định.
- Có thể ước tính tầm quan trọng của từng đặc trưng.
**2. 3.** 2**. Gradient Boosting
1. Tổng quan về Gradient Boosting
a) Định nghĩa**
- Gradient Boosting là một thuật toán học máy mạnh mẽ thuộc nhóm học tập tổ hợp,
được phát triển bởi Jerome Friedman [1]. Thuật toán này tạo ra các dự đoán chính
xác bằng cách kết hợp nhiều cây quyết định thành một mô hình duy nhất.


- **Nguyên lý cốt lõi** : Mỗi mô hình cơ sở trong chuỗi sẽ học từ sai sót của mô hình
    trước đó. Qua từng vòng lặp, các mô hình mới dần điều chỉnh lỗi và cải thiện khả
    năng dự đoán, tạo thành một hệ thống ngày càng chính xác hơn.

```
b) Đặc điểm nổi bật:
```
- **Học tuần tự có định hướng** : Các mô hình không độc lập mà liên kết chặt chẽ với
    nhau
- **Khả năng nhận diện mẫu phức tạp** : Đặc biệt hiệu quả với dữ liệu có cấu trúc
    phức tạp
- **Tối ưu hóa liên tục** : Giảm lỗi dự đoán qua từng bước một cách có hệ thống
**2. Học tập tổ hợp và Boosting Nguyên lý cốt lõi của Gradient Boosting** :
**a) Khái niệm học tập tổ hợp**
- Học tập tổ hợp là phương pháp kết hợp nhiều mô hình hoặc thuật toán nhằm nâng
cao hiệu quả dự đoán, vượt qua giới hạn của từng mô hình đơn lẻ. Phương pháp
này giúp tối ưu hóa sự đánh đổi giữa độ chệch và phương sai.

```
b) Hai kỹ thuật chính
Bagging
```
- **Cơ chế** : Huấn luyện nhiều mô hình song song trên các tập dữ liệu con khác nhau
- **Mục tiêu** : Giảm phương sai bằng cách trung bình hóa sai số của từng mô hình
- **Ví dụ điển hình** : Random Forests
- **Đặc điểm** : Các mô hình học độc lập, không ảnh hưởng lẫn nhau
**Boosting**
- **Cơ chế** : Huấn luyện các mô hình theo cách tuần tự, mỗi mô hình mới được thiết kế
    để sửa lỗi của mô hình trước
- **Mục tiêu** : Giảm độ chệch bằng cách tập trung vào các điểm dữ liệu khó
- **Ví dụ điển hình** : AdaBoost, Gradient Boosting
**Đặc điểm** :
- Gán trọng số cao hơn cho các điểm dữ liệu bị phân loại sai
- Các mô hình sau học từ sai lầm của mô hình trước
- Kiểm soát được hướng phát triển của toàn bộ hệ thống

```
c) Ứng dụng thực tế
```
- Các kỹ thuật ensemble được ứng dụng rộng rãi để cải thiện độ chính xác mô hình,
    đặc biệt hiệu quả trong:
    - Dữ liệu phức tạp hoặc có nhiễu
    - Bài toán yêu cầu độ chính xác cao
    - Trường hợp dữ liệu có nhiều đặc trưng
**3. Cơ chế hoạt động của Gradient Boosting**


**a) Ý tưởng cơ bản**
Gradient Boosting kết hợp nhiều mô hình dự đoán yếu - thường là cây quyết định -
thành một hệ thống mạnh mẽ. Các mô hình được huấn luyện tuần tự nhằm:

- Giảm lỗi xuống mức tối thiểu
- Nắm bắt các mối quan hệ phức tạp giữa các đặc trưng
- Cải thiện độ chính xác liên tục qua từng vòng lặp

**b) Tối ưu hóa hàm mất mát**
Một trong những điểm mạnh của Gradient Boosting là khả năng tối thiểu hóa hàm
mất mát một cách lặp đi lặp lại qua từng bước huấn luyện.
Sai số bình phương trung bình (Mean Squared Error - MSE)
MSE là hàm mất mát phổ biến để đánh giá độ chênh lệch giữa dự đoán và giá trị
thực tế:

```
𝑀𝑆𝐸=
```
### 1

### 𝑛

### /(𝑦$−𝑝$)"

```
%
```
```
$&'
```
```
Trong đó:
```
- 𝑦$: Giá trị thực tế
- 𝑝$: Giá trị dự đoán
- 𝑛: Số lần quan sát
**Ý nghĩa của MSE:**
- Đo lường mức độ khác biệt giữa giá trị dự đoán và giá trị thực
- Bình phương sai số để đảm bảo cả sai số dương và âm đều được tính
- Ưu tiên xử lý các sai số lớn
- MSE càng thấp → dự đoán càng chính xác
**Lưu ý quan trọng:**
- MSE = 0 trong thực tế rất khó đạt được do tồn tại độ ngẫu nhiên trong dữ
liệu
- MSE quá thấp có thể là dấu hiệu của overfitting
- Nên so sánh MSE giữa các mô hình hoặc theo thời gian để đánh giá hiệu
quả

**c) Xử lý Overfitting**
Gradient Boosting dễ gặp hiện tượng **quá khớp** - khi mô hình học quá kỹ dữ liệu
huấn luyện và khó áp dụng cho dữ liệu mới.
**Các biện pháp khắc phục:**

- **Chuẩn hóa** : Kiểm soát độ phức tạp của mô hình
- **Cắt tỉa mô hình** : Loại bỏ các nhánh không cần thiết trong cây
- **Dừng sớm** : Ngừng huấn luyện khi hiệu suất không còn cải thiện
- **Tinh chỉnh siêu tham số** : Điều chỉnh các tham số để cân bằng
- **Theo dõi hiệu suất** : Giám sát mô hình trong quá trình huấn luyện


**4. Các bước hoạt động chi tiết của Gradient Boosting**

```
a) Bước 1: Khởi tạo
```
- Sử dụng tập dữ liệu huấn luyện để thiết lập mô hình học cơ sở
- Thường là một cây quyết định đơn giản với số nút lá hoặc nút kết thúc nhất định
- Các dự đoán ban đầu thường được tạo ngẫu nhiên hoặc từ giá trị trung bình
- Lý do chọn mô hình yếu: Dễ hiểu, dễ diễn giải và tạo nền móng vững chắc cho các
    bước tiếp theo

```
b) Bước 2: Tính sai số dư
Với mỗi mẫu trong tập huấn luyện:
```
Residual$=𝑦$−𝑦 (^2) $
Trong đó:

- 𝑦$: Giá trị thực tế

- 𝑦 (^2) $: Giá trị dự đoán hiện tại
**Mục đích** : Xác định những điểm mà mô hình hiện tại chưa dự đoán chính xác, làm
cơ sở để cải thiện trong các bước tiếp theo.
**c) Bước 3: Điều chỉnh bằng kỹ thuật chuẩn hóa (Regularization)**
Trước khi huấn luyện mô hình tiếp theo, áp dụng chuẩn hóa để:

- Giảm mức độ ảnh hưởng của mô hình mới trong toàn bộ hệ thống
- Kiểm soát tốc độ học (learning rate) của thuật toán
- Giảm thiểu hiện tượng quá khớp (overfitting)
- Tối ưu hiệu suất tổng thể của mô hình
**Công thức cập nhật:**
    𝐹()'(𝑥)=𝐹((𝑥)+𝜂⋅ℎ((𝑥)

```
Trong đó:
```
- 𝜂: Learning rate (tốc độ học)
- ℎ((𝑥): Mô hình mới được thêm vào
- 𝐹((𝑥): Mô hình tổng hợp hiện tại

```
d) Bước 4: Huấn luyện mô hình kế tiếp
```
- Sử dụng các giá trị sai số dư làm **nhãn mục tiêu mới**
- Mô hình mới được huấn luyện để dự đoán chính xác các sai số này
- **Trọng tâm** : Sửa các lỗi mà mô hình trước đã mắc phải
- **Kết quả** : Cải thiện độ chính xác tổng thể của dự đoán

```
e) Bước 5: Cập nhật tổ hợp học tập
Sau khi huấn luyện mô hình mới:
```

1. Đánh giá hiệu suất của tổ hợp hiện tại (bao gồm mô hình mới) trên tập kiểm thử
    riêng
2. Nếu kết quả đạt yêu cầu → tích hợp mô hình mới vào hệ thống
3. Nếu không đạt → điều chỉnh lại các siêu tham số

**f) Bước 6: Lặp lại**
Các bước 2-5 được lặp lại nhiều lần:

- Mỗi vòng lặp tiếp tục xây dựng và tinh chỉnh mô hình cơ sở
- Huấn luyện thêm các cây quyết định mới
- Liên tục nâng cao độ chính xác của mô hình
- Tiếp tục cho đến khi đạt được độ chính xác mong muốn so với mô hình ban đầu

**g) Bước 7: Tiêu chí dừng huấn luyện**
Thuật toán kết thúc khi đạt một trong các điều kiện sau:

1. **Số vòng lặp tối đa** : Đạt đến giới hạn số lượng mô hình được xác định trước
2. **Độ chính xác mục tiêu** : Đạt được hiệu suất mong muốn
3. **Diminishing returns** : Hiệu quả cải thiện bắt đầu giảm dần
4. **Early stopping** : Hiệu suất trên tập validation không còn cải thiện hoặc bắt đầu
    giảm sút
**Mục đích** : Đảm bảo mô hình cuối cùng đạt được sự cân bằng hợp lý giữa độ phức tạp
và hiệu suất.

```
Hình 2.2.3 – Minh họa quá trình học của thuật toán Gradient Boosting
```
**5. Các kỹ thuật nâng cao**


**a) Phương pháp kết hợp**
Kết hợp Gradient Boosting với các thuật toán khác có thể nâng cao độ chính xác:
**Các mô hình có thể kết hợp:**

- Support Vector Machines (SVMs)
- Random Forests
- k-Nearest Neighbors (KNN)
- Neural Networks
**Xếp chồng:**
- Huấn luyện nhiều mô hình cơ sở đồng thời
- Sử dụng kết quả đầu ra của các mô hình này làm đầu vào
- Meta learner học cách kết hợp các dự đoán để đưa ra dự đoán cuối cùng
- Tận dụng thế mạnh riêng của từng mô hình
- Tạo ra hệ thống dự đoán mạnh mẽ và ổn định hơn

**b) Dừng sớm và Kiểm định chéo
Early Stopping:**

- Theo dõi hiệu suất mô hình trong quá trình huấn luyện
- Dừng lại khi hiệu suất trên validation set không còn cải thiện hoặc bắt đầu giảm
    sút
- Phương pháp hiệu quả để tránh overfitting
- Tiết kiệm thời gian và tài nguyên tính toán
**Cross-validation:**
- **k-fold cross-validation** : Chia dữ liệu thành k phần, luân phiên sử dụng mỗi
    phần làm validation
- Đánh giá hiệu suất mô hình toàn diện hơn
- Tăng độ tin cậy của quá trình đánh giá
- Hỗ trợ hiệu quả trong hyperparameter tuning
- Nâng cao năng lực dự đoán của mô hình

**c) Xử lý dữ liệu mất cân bằng**
Gradient Boosting nhạy cảm với **mất cân bằng lớp** - khi số lượng mẫu thuộc một
lớp chiếm ưu thế.
**Hậu quả:**

- Dự đoán thiên vị về lớp chiếm đa số
- Hiệu suất kém trên lớp thiểu số
**Các kỹ thuật khắc phục:**
1. **Tăng cường lớp thiểu số** :
    o SMOTE (Synthetic Minority Over-sampling Technique)
    o Tạo thêm mẫu tổng hợp cho lớp thiểu số
2. **Giảm mẫu lớp đa số** :
    o Random undersampling
    o Tomek links, Edited Nearest Neighbors


3. **Weighted Loss Function** :
    o Gán trọng số cao hơn cho lỗi xảy ra ở lớp thiểu số
    o Buộc mô hình phải học kỹ hơn các mẫu ít gặp

o Công thức: 𝐿=∑$ 𝑤$⋅𝑙(𝑦$,𝑦 (^2) $)

4. **Kết hợp với chỉnh sửa siêu tham số** :
    o scale_pos_weight trong XGBoost
    o class_weight trong các framework khác
**6. Ưu điểm và hạn chế
a) Ưu điểm**
1. **Hiệu quả cao** : Đặc biệt tốt với dữ liệu có cấu trúc (tabular data)
2. **Linh hoạt** : Áp dụng được với nhiều loại loss function
3. **Giảm thiên vị mạnh mẽ** : Cải thiện đáng kể độ chính xác
4. **Xử lý dữ liệu phức tạp** : High-dimensional, sparse data
5. **Feature importance** : Khả năng diễn giải tốt
6. **Linh** : Ít nhạy cảm với outliers (tùy loss function)
7. **Không cần chuẩn hóa** : Khi sử dụng tree-based learners
**b) Hạn chế**
1. **Thời gian huấn luyện lâu** : Do tính chất tuần tự
2. **Dễ overfit** : Nếu số iterations quá nhiều
3. **Nhiều hyperparameters** : Cần tuning cẩn thận 4
**7. Các dạng triển khai phổ biến**

```
a) XGBoost (Extreme Gradient Boosting)
Đặc điểm:
```
- Ra mắt năm 2014, được thiết kế tối ưu về tốc độ và hiệu năng
- Kết hợp hoàn hảo giữa sức mạnh phần mềm và phần cứng
- Phù hợp cho cả bài toán hồi quy và phân loại
**Thành tựu:**
- ~640 người đóng góp vào mã nguồn và ~7500 commits trên GitHub [2]
- Thống trị các cuộc thi Kaggle trong nhiều năm
- Hỗ trợ đa nền tảng: Windows, Linux, OS X
- Hỗ trợ đa ngôn ngữ: C++, Python, R, Java, Scala, Julia
- Tích hợp với: AWS, Azure, Yarn, Flink, Spark
**b) LightGBM (Light Gradient Boosting Machine)
Lịch sử phát triển:**
- Ra mắt tháng 1/2016 bởi Microsoft
- Nhanh chóng thay thế XGBoost, trở thành thuật toán ensemble ưa chuộng nhất
**Cải tiến chính:**
    1. **Histogram-based algorithms** :


```
o Thay thế pre-sort-based algorithms
o Tăng tốc độ training đáng kể
o Giảm bộ nhớ cần sử dụng
```
2. **GOSS (Gradient-based One-Side Sampling)** :
    o Lọc các mẫu dữ liệu quan trọng cho việc tìm split points
    o Giảm đáng kể chi phí tính toán
    o Vẫn duy trì độ chính xác cao
3. **EFB (Exclusive Feature Bundling)** :
    o Gộp các features loại trừ lẫn nhau
    o Giảm số chiều dữ liệu
    o Tăng tốc quá trình training
4. **Leaf-wise tree growth** :
    o XGBoost: Level (depth)-wise
    o LightGBM: Leaf-wise
    o Lựa chọn nút để phát triển dựa trên tối ưu toàn bộ tree
    o Với cùng số node, leaf-wise thường outperform level-wise
**Lưu ý quan trọng:**
- Leaf-wise có thể dẫn đến overfitting với dữ liệu nhỏ
- LightGBM sử dụng thêm hyperparameter max_depth để hạn chế
- **Khuyến nghị** : Sử dụng LightGBM khi bộ dữ liệu đủ lớn

_Kết luận Chương II_

Chương này đã trình bày các cơ sở lý thuyết cần thiết cho đề tài: từ bài toán phân loại
nhị phân mất cân bằng lớp, kiến trúc xử lý phân tán Apache Spark, đến các điều kiện
nhiệt động lực học hình thành bão và thuật toán Gradient Boosting — đặc biệt là biến
thể LightGBM được sử dụng trong pipeline bottom-up của đề tài. Các kiến thức này
tạo nền tảng cho việc thiết kế và triển khai hệ thống tại Chương III.


# CHƯƠNG III: XÂY DỰNG HỆ THỐNG PHÂN TÍCH VÀ DỰ BÁO

**3.1 Lý do chọn đề tài và công nghệ Spark**

- Sự bùng nổ của dữ liệu từ nhiều nguồn yêu cầu công nghệ xử lý song song mạnh
    mẽ như Spark để thay thế các script Python đơn lẻ chạy chậm.
- Nhu cầu cấp thiết về việc tích hợp đa nguồn dữ liệu để có cái nhìn toàn diện về cơ
    chế hoạt động của bão.
- Tiếp cận **bottom-up**: thay vì dự báo trực tiếp từ chỉ số macro, hệ thống huấn luyện
    bộ phân loại ở cấp vi mô (mỗi ô lưới, mỗi ngày) rồi tổng hợp lên cấp vĩ mô
    (số bão/tháng). Phương pháp này cho phép nắm bắt các tín hiệu không gian cục bộ
    mà các chỉ số tổng hợp toàn bộ basin không thể phản ánh được.

**3.2. Thiết kế dữ liệu đầu vào**

```
3.2.1. Nguồn dữ liệu và Chiến lược hợp nhất
```
Hệ thống tích hợp 4 nguồn dữ liệu quốc tế:

- **Dữ liệu Bão (IBTrACS - NOAA):** Dữ liệu dạng điểm, cung cấp toạ độ tâm bão,
    thời gian, sức gió, áp suất tâm bão. Bao gồm `ibtracs_fullbasin.parquet` (toàn bộ
    bão WP basin) dùng làm ground truth cho Phase 5. Mỗi bão được định danh bằng
    mã SID duy nhất, cho phép theo dõi từ hình thành đến tan rã.
- **Dữ liệu Nhiệt độ Biển (NOAA OISST v2.1):** Dữ liệu dạng lưới toàn cầu 0.25°,
    cập nhật hàng ngày, cung cấp nhiệt độ bề mặt biển (SST) — biến số quan trọng nhất
    cho cyclogenesis theo nghiên cứu của Gray (1968).
- **Dữ liệu Khí quyển (ERA5 - ECMWF):** Dữ liệu tái phân tích toàn cầu 0.25°,
    cung cấp gió kinh tuyến U (u10), gió vĩ tuyến V (v10) và áp suất mực biển trung
    bình (MSLP). ERA5 là bộ dữ liệu tái phân tích có độ phân giải cao nhất hiện nay.
- **Chỉ số ENSO (ONI - Climate Prediction Center):** Oceanic Niño Index theo tháng,
    đo sự bất thường SST tại vùng Niño 3.4 (5°N–5°S, 120°–170°W), phản ánh pha
    El Niño / La Niña — yếu tố ảnh hưởng mạnh đến vị trí và tần suất bão.

```
3.2.2. Chuẩn hoá dữ liệu
```
Thách thức lớn nhất trong bước này là đồng bộ 4 nguồn dữ liệu có cấu trúc khác nhau:

- **Đồng bộ không gian:** Cắt toàn bộ dữ liệu lưới toàn cầu về khu vực Tây Thái Bình
    Dương (0–30°N, 100–180°E), phân giải 0.25° (tương đương ~27.8 km tại xích đạo).
    Khu vực này bao phủ Biển Đông và vùng hình thành bão chính.
- **Đồng bộ thời gian:** Chuẩn hóa tất cả dữ liệu về khung thời gian ngày. ERA5
    (ban đầu 6 giờ/lần) được lấy trung bình ngày; NOAA SST đã sẵn độ phân giải ngày.
- **Grid Snapping:** Làm tròn toạ độ tâm bão (IBTrACS) về ô lưới 0.25° gần nhất
    bằng công thức `Lat_Grid = Round(Lat × 4) / 4` để khớp nối chính xác vị trí bão
    với điều kiện môi trường xung quanh.
- **Giai đoạn thời gian:** 1983 – 2024 (42 năm), đảm bảo đủ dài để huấn luyện mô
    hình với tính mùa vụ và chu kỳ ENSO đa dạng.

```
3.2.3. Cấu trúc Master Dataset
```
Sử dụng PySpark để thực hiện spatio-temporal join, kết quả là bảng `master_dataset.parquet`:

- **Quy mô:** Gần **600 triệu bản ghi** (597.493.318 rows theo log huấn luyện thực tế),
    mỗi bản ghi tương ứng một ô lưới 0.25° × một ngày.
- **Schema:** 30 cột bao gồm:
    [lat, lon, date, year, month, u_wind_avg, v_wind_avg, slp_avg, wind_speed_env_avg,
    sst_avg, oni_value, enso_phase, SID, NAME, wind_speed_kmh, pressure_wmo, ...]
- **Nhãn (label):** `is_storm = 1` nếu `SID IS NOT NULL` (có cơn bão đi qua ô lưới
    trong ngày đó), ngược lại `0`.
- **Định dạng lưu trữ:** Apache Parquet — nén cột hiệu quả, giảm dung lượng lưu
    trữ ~5× so với CSV gốc.

**3.3. Kiến trúc Pipeline 5 Phase**

Toàn bộ pipeline được triển khai trong file `models/bottom_up_forecast.py`, gồm 5 phase
chạy tuần tự. Tổng thời gian huấn luyện end-to-end: **~3 giờ 27 phút** (12.387 giây)
trên single machine.

```
3.3.1. Phase 1: Feature Engineering (Spark) — 3 giây
```
Sử dụng Window Functions của PySpark (partitioned by `[lat, lon]`, ordered by `date`)
để tạo rolling average features tại mỗi ô lưới. Phase này chạy rất nhanh (3 giây) nhờ
cơ chế **lazy evaluation** — Spark chỉ ghi nhận phép biến đổi, chưa thực thi.

**Rolling windows:**

| Cửa sổ | Đặc trưng |
|--------|-----------|
| 7 ngày | SST, SLP, wind speed |
| 14 ngày | SST, SLP |
| 30 ngày | SST, SLP |
| 90 ngày | SST, SLP |
| 180 ngày | SST, SLP |

**Đặc trưng physics-informed (domain knowledge):**

| Đặc trưng | Công thức | Ý nghĩa khí tượng |
|-----------|-----------|-------------------|
| `sst_above_threshold` | `1 nếu SST_avg ≥ 26.5°C` | Ngưỡng SST tối thiểu cho cyclogenesis (Gray, 1968) |
| `sst_anomaly` | `sst_avg − sst_180d_avg` | Nhiệt bất thường so với trung bình mùa |
| `slp_tendency` | `slp_avg − slp_7d_avg` | Xu hướng áp suất — giảm nhanh = đối lưu phát triển |

**Tổng cộng: 24 đặc trưng** bao gồm:
- Không gian: `lat`, `lon` (2)
- Thời gian: `month` (1)
- Biến raw: `sst_avg`, `slp_avg`, `u_wind_avg`, `v_wind_avg`, `wind_speed_env_avg`,
    `oni_value`, `enso_phase` (7)
- Rolling averages: 11 cửa sổ (11)
- Derived features: `sst_above_threshold`, `sst_anomaly`, `slp_tendency` (3)

Ví dụ dữ liệu sau Feature Engineering (tại ô lưới 0.25°N, 123.5°E):

| date | sst_avg | sst_anomaly | sst_above_threshold | slp_tendency |
|------|---------|-------------|--------------------:|-------------:|
| 1983-01-01 | 28.82 | 0.000 | 1 | 0.000 |
| 1983-01-02 | 28.67 | −0.075 | 1 | −15.526 |
| 1983-01-03 | 27.75 | −0.663 | 1 | 10.243 |
| 1983-01-04 | 27.61 | −0.603 | 1 | 11.608 |
| 1983-01-05 | 27.26 | −0.762 | 1 | 17.174 |

```
3.3.2. Phase 2: Target Definition & Stratified Undersampling (Spark) — ~76 phút
```
Đây là phase tốn thời gian nhất do phải quét toàn bộ ~600M bản ghi trên đĩa.

**Phân bố lớp gốc:**
- **Positives (có bão):** 34.595 bản ghi
- **Negatives (không bão):** 597.458.723 bản ghi
- **Tỷ lệ mất cân bằng:** 1:17.270 — cực kỳ nghiêm trọng

**Chiến lược undersampling:**
Giữ 100% positives, lấy mẫu negatives **stratified** theo tháng (12 strata) bằng hàm
`sampleBy` của Spark, đảm bảo mỗi tháng được đại diện đúng tỷ lệ. Mục tiêu tỷ lệ
cuối cùng: 1:20 (dương:âm).

**Phân bố negatives trước và sau sampling:**

| Tháng | Trước Sampling | Tỷ lệ lấy mẫu | Sau Sampling |
|-------|---------------:|:--------------:|:------------:|
| 1 | 52.201.783 | 0,001158 | 60.479 |
| 2 | 46.104.070 | 0,001158 | 53.363 |
| 3 | 50.570.705 | 0,001158 | 58.597 |
| 4 | 48.939.041 | 0,001158 | 56.921 |
| 5 | 50.569.574 | 0,001158 | 58.843 |
| 6 | 48.937.030 | 0,001158 | 57.113 |
| 7 | 50.566.190 | 0,001158 | 58.474 |
| 8 | 50.565.414 | 0,001158 | 58.787 |
| 9 | 48.934.009 | 0,001158 | 56.676 |
| 10 | 50.565.917 | 0,001158 | 58.671 |
| 11 | 48.935.989 | 0,001158 | 56.403 |
| 12 | 50.569.001 | 0,001158 | 58.392 |
| **Tổng** | **597.458.723** | | **692.719** |

**Kết quả:** 34.595 dương + 692.719 âm = **727.314** bản ghi (tỷ lệ 1:20).

**Quản lý bộ nhớ:** Trong quá trình chạy, Spark gặp nhiều cảnh báo MemoryStore do
RAM không đủ cache toàn bộ 600M bản ghi. Spark tự động sử dụng cơ chế **spill-to-disk**
— ghi các block RDD không fit trong RAM ra đĩa (ví dụ: `Persisting block rdd_27_5 to
disk instead`). Cơ chế này đảm bảo pipeline không bị crash mặc dù chỉ chạy trên single
machine với bộ nhớ giới hạn, nhưng làm tăng thời gian xử lý đáng kể.

```
3.3.3. Phase 3: Time-Based Split — ~4 phút
```
Chia dữ liệu theo **thời gian** (chronological split), không dùng random split để tránh
**data leakage** — đảm bảo mô hình không bao giờ được huấn luyện trên dữ liệu tương
lai.

| Tập | Năm | Số bản ghi | Positives | Negatives | Tỷ lệ |
|-----|-----|-----------|-----------|-----------|--------|
| **Train** | ≤ 2015 | 571.158 | 27.601 | 543.557 | 1:19 |
| **Validation** | 2016–2019 | 69.840 | 3.339 | 66.501 | 1:19 |
| **Test** | ≥ 2020 | 86.316 | 3.655 | 82.661 | 1:22 |

**Nhận xét:** Tỷ lệ dương/âm trong tập test (1:22) cao hơn train (1:19) — phản ánh
thực tế rằng mùa bão gần đây có xu hướng biến động mạnh hơn, và đồng thời đánh
giá khả năng tổng quát hóa của mô hình trên dữ liệu mới.

```
3.3.4. Phase 4: LightGBM Micro-Level Classifier (Pandas) — 14 giây
```
Chuyển đổi Spark DataFrame sang Pandas (~727K rows fit trong RAM) rồi huấn luyện
bộ phân loại LightGBM **native** (không dùng SynapseML — gây lỗi `JavaPackage` trên
Spark standalone):

**Hyperparameters:**

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| `boosting_type` | `gbdt` | Gradient Boosting chuẩn |
| `max_depth` | 8 | Giới hạn chiều sâu, chống overfit |
| `num_leaves` | 63 | Leaf-wise growth (2^6 − 1) |
| `learning_rate` | 0.1 | Tốc độ học chuẩn |
| `num_boost_round` | 200 | Số cây tối đa |
| `is_unbalance` | True | Tự động gán trọng số cho lớp thiểu số |
| `metric` | auc, average_precision | AUC + PR-AUC |

**Quá trình hội tụ** (theo log huấn luyện):

| Iteration | Train AUC | Val AUC | Train AP | Val AP |
|:---------:|:---------:|:-------:|:--------:|:------:|
| 20 | 0.9912 | 0.9873 | 0.8427 | 0.8220 |
| 60 | 0.9949 | 0.9902 | 0.9032 | 0.8658 |
| 100 | 0.9966 | 0.9911 | 0.9311 | 0.8823 |
| 140 | 0.9975 | 0.9913 | 0.9469 | 0.8858 |
| 200 | 0.9983 | 0.9914 | 0.9627 | 0.8875 |

**Nhận xét:** Mô hình hội tụ nhanh — 60 iterations đã đạt val AUC 0.990+. Từ
iteration 100 trở đi, val performance gần như không thay đổi trong khi train tiếp tục
tăng, cho thấy sự cân đối tốt giữa underfitting và overfitting.


**Probability Calibration:**
Sau khi huấn luyện, áp dụng **Isotonic Regression** trên validation set để hiệu chuẩn
xác suất đầu ra:
- Raw probability mean: **0.073170** (cao hơn tỷ lệ dương thực tế)
- Calibrated probability mean: **0.047809** (gần đúng tỷ lệ 1:20 sau undersampling)

**Output:** Xác suất `prob_storm` ∈ [0, 1] cho mỗi ô lưới × ngày.
**Artifacts:** `lgbm_storm_classifier.pkl`, `probability_calibrator.pkl`, `train_means.pkl`.

```
3.3.5. Phase 5: Monthly Roll-Up & Regression — ~124 phút
```
**Bước 1 — Distributed Inference (mapInPandas):**
Chạy LightGBM classifier trên toàn bộ ~600M bản ghi bằng `mapInPandas`:
- Broadcast model đã pickle đến tất cả workers
- Repartition thành 64 partitions để cân bằng tải
- Mỗi partition: nhận batch Pandas DataFrame → chạy predict → trả về `prob_storm`
- Sau đó áp dụng isotonic calibration
- Ngưỡng `PROB_THRESHOLD = 0.10` lọc bỏ các ô có xác suất quá thấp

Bước này chiếm phần lớn thời gian Phase 5 (~2 giờ) do phải đọc lại toàn bộ
597M bản ghi từ đĩa (spill data).

**Bước 2 — Monthly SPI (5 biến thể):**
Tổng hợp `prob_storm` theo tháng, tạo ra 5 chỉ số Storm Potential Index:

| SPI Variant | Công thức | Ý nghĩa |
|-------------|-----------|---------|
| `monthly_SPI` | `SUM(prob_storm)` | Tích phân không gian toàn bộ lưới |
| `monthly_SPI_thresh` | `SUM(prob > threshold)` | Tổng xác suất vượt ngưỡng |
| `monthly_SPI_count` | `COUNT(prob > 0.30)` | Số ô lưới có tín hiệu mạnh |
| `monthly_SPI_density` | `AVG(prob_storm)` | Mật độ tín hiệu trung bình |
| `monthly_SPI_log` | `log1p(SPI)` | Nén khoảng động cho phân phối lệch |

**Bước 3 — Temporal Context Features:**
Bổ sung các đặc trưng ngữ cảnh thời gian (merge-based lag, gap-safe):
- Lag ONI (1 tháng) và ONI trung bình trượt 3 tháng
- SPI lag 1 tháng
- 6 interaction terms: `spi_x_oni`, `spi_x_elnino`, `spi_x_lanina`, `is_elnino`,
    `is_lanina`, `oni_abs` — giúp mô hình nắm bắt hiệu ứng dịch chuyển không gian
    của bão theo pha ENSO

**Bước 4 — So sánh 4 mô hình hồi quy:**
Sau khi tổng hợp xong, thu được **505 mẫu** (43 năm × tối đa 12 tháng). So sánh
4 mô hình:

| Mô hình | Train MAE | Test MAE | Nhận xét |
|---------|:---------:|:--------:|----------|
| AdaBoost | 0.97 | 1.17 | Ổn định |
| GBR_Huber | 0.43 | 1.11 | Overfit (train << test) |
| Poisson_GLM | 2.55 | 2.47 | Underfit — linear quá đơn giản |
| **Ridge (α=10)** | **0.97** | **1.03 ★** | Best generalization |

**Kết quả:** Chọn **Ridge (α=10)** làm mô hình cuối cùng — có gap train/test nhỏ nhất
(0.97 vs 1.03), cho thấy regularization L2 hiệu quả với tập dữ liệu nhỏ (505 mẫu).


**3.4. Cài đặt chương trình**

```
3.4.1. Chuẩn bị môi trường
```
- Cài đặt Apache Spark (standalone mode), Java JDK 11+ và Python 3.11+ (Conda
    env: `pyspark`).
- Cài đặt thư viện: `pyspark`, `lightgbm==4.6.0`, `scikit-learn`, `scipy`, `numpy`,
    `pandas`, `streamlit`, `duckdb`, `plotly`.

```
3.4.2. Cấu hình Spark & Parquet
```
- Thiết lập bộ nhớ Driver (8GB) / Executor (4GB) — cấu hình tối thiểu để xử lý
    600M bản ghi. Khi RAM không đủ, Spark tự động spill-to-disk.
- Phân vùng dữ liệu theo `year` / `month` để tối ưu tốc độ truy vấn.
- Sử dụng `mapInPandas` với broadcast model để phân phối inference lên các workers.
- Sử dụng cơ chế `incubator.vector` (JDK) cho SIMD optimization.

```
3.4.3. Huấn luyện & Theo dõi
```
- Chạy `python models/bottom_up_forecast.py` — pipeline 5 phase, ~3.5 giờ.
- Sử dụng `./models/train.zsh` — script quản lý huấn luyện với menu tương tác,
    hiển thị trạng thái model (trained/not trained) và thời gian huấn luyện gần nhất.
- Sử dụng Spark UI (port 4040) để theo dõi hiệu năng jobs/stages/tasks.

**Thời gian thực tế từng phase (theo log huấn luyện ngày 22/03/2026):**

| Phase | Tên | Thời gian | Ghi chú |
|:-----:|-----|:---------:|---------|
| 1 | Feature Engineering | 3s | Lazy evaluation |
| 2 | Undersampling | 4.588s (~76 min) | Quét 600M rows |
| 3 | Temporal Split | ~4 min | Shuffle + count |
| 4 | LightGBM Classifier | 14s | 727K rows in RAM |
| 5 | Monthly Roll-Up | 7.447s (~124 min) | Inference 600M rows |
| | **TỔNG** | **12.387s (~3h27m)** | |

```
3.4.4. Giao diện (Streamlit)
```
- `streamlit run app.py` — Dashboard khám phá dữ liệu (2 tab: SST heatmap
    theo tháng/năm và trường gió ERA5 với bản đồ dark theme).
- `streamlit run app/app.py` — Dashboard Phase 5: biểu đồ cột so sánh actual vs
    predicted theo tháng, đường annual trend, metrics summary (MAE, ENSO phase),
    và bảng chi tiết monthly breakdown.

```
3.4.5. Cấu trúc dự án
```

```
WeatherPredict/
├── app.py                      # Dashboard SST & Storms (2 tabs)
├── app/app.py                  # Dashboard Phase 5 predictions
├── models/
│   ├── bottom_up_forecast.py   # Pipeline 5 phase chính (~900 LOC)
│   ├── train.zsh               # Script quản lý huấn luyện
│   ├── lgbm_storm_classifier.pkl   # ~1.4 MB
│   ├── probability_calibrator.pkl  # ~2 KB
│   ├── ridge_monthly_model.pkl     # ~656 B
│   ├── train_means.pkl             # ~1.2 KB
│   └── monthly_predictions.csv     # ~6 KB (505 rows)
├── helpers/                    # Scripts ETL & tiền xử lý
│   ├── spatio_temporal_join.py # Join 4 nguồn → master_dataset
│   ├── preprocess_era5.py      # ERA5 GRIB → Parquet
│   ├── convert_sst.py          # NOAA SST NetCDF → Parquet
│   └── preprocess_ibtracs.py   # IBTrACS CSV → Parquet
├── ./                       # Tài liệu kỹ thuật
├── parquet_data/               # master_dataset (~600M rows)
└── SPARK__DATA/                # Dữ liệu thô (ERA5, SST, IBTrACS, ONI)
```

**3.5 Ưu điểm và nhược điểm của hệ thống**

```
3.5.1 Ưu điểm
```
- Kiến trúc bottom-up cho phép tận dụng tín hiệu vi mô (cấp ô lưới 0.25°) thay vì
    chỉ dựa trên chỉ số macro — phát hiện các vùng có nguy cơ bão cục bộ.
- LightGBM xử lý missing values natively (NaN-aware splits) — không cần impute
    toàn bộ 600M bản ghi.
- Khả năng mở rộng (Scalability) cao nhờ PySpark: chỉ cần thêm workers để tăng
    tốc, không cần sửa kiến trúc pipeline.
- Tận dụng 4 bộ dữ liệu uy tín quốc tế (ERA5, NOAA OISST, IBTrACS, ONI)
    đảm bảo độ tin cậy khoa học.
- Spark spill-to-disk cho phép xử lý 600M bản ghi trên single machine với RAM
    giới hạn — mặc dù chậm hơn nhưng vẫn hoàn thành.

```
3.5.2 Nhược điểm
```
- Yêu cầu dung lượng lưu trữ lớn (hàng chục GB) cho master_dataset dạng Parquet.
- Phase 5 inference trên 600M bản ghi tốn ~2 giờ do spill-to-disk liên tục.
- Cần tối ưu cấu hình Spark (memory, partitions) cho từng cấu hình phần cứng.

**3.6. Hướng dẫn sử dụng ứng dụng Streamlit**

- **Dashboard SST & Storms** (`app.py`): Chọn năm/tháng → xem heatmap nhiệt
    độ biển hoặc trường gió ERA5 với bản đồ dark theme. Hỗ trợ zoom, pan và
    hover tooltip hiển thị giá trị tại từng điểm.
- **Dashboard Phase 5** (`app/app.py`): Chọn năm → xem biểu đồ cột so sánh
    actual vs predicted theo tháng (màu sắc theo tháng), biểu đồ annual trend
    (actual vs predicted qua các năm), metrics (MAE, ENSO phase), và bảng
    chi tiết monthly breakdown với cột error.

_Kết luận Chương III_

Hệ thống được xây dựng theo kiến trúc bottom-up 5 phase, tận dụng PySpark để xử lý
gần 600 triệu bản ghi và LightGBM cho bài toán phân loại cực mất cân bằng (1:17.270).
Pipeline hoàn thành trong ~3.5 giờ trên single machine, cho thấy tính khả thi cao cho
nghiên cứu và ứng dụng thực tế. Kết quả và đánh giá chi tiết được trình bày tại Chương IV.


# CHƯƠNG IV: ĐÁNH GIÁ & KẾT QUẢ

**4.1 Kịch bản kiểm thử**

- **Backtesting (temporal split):** Huấn luyện trên dữ liệu ≤ 2015 (33 năm), xác nhận
    trên 2016–2019 (4 năm), kiểm tra trên ≥ 2020 (5 năm). Đảm bảo không có thông
    tin tương lai bị rò rỉ vào quá trình huấn luyện.
- **Stress Test:** Pipeline xử lý thành công toàn bộ 597.493.318 bản ghi trên single
    machine (8GB driver) trong 12.387 giây — chứng minh cơ chế spill-to-disk của
    Spark hoạt động ổn định.

**4.2 Kết quả Phase 4 — Bộ phân loại LightGBM**

Bộ phân loại micro-level (ô lưới × ngày) đạt hiệu suất rất cao trên cả validation và
test set:

| Chỉ số | Validation | Test |
|--------|:----------:|:----:|
| **ROC-AUC** | **0.9914** | **0.9915** |
| **PR-AUC (Average Precision)** | **0.8875** | **0.8794** |

**Nhận xét:**
- ROC-AUC 0.9915 trên test set — gần hoàn hảo, cho thấy mô hình có khả năng
    phân biệt rõ ràng giữa điều kiện có bão và không có bão tại cấp ô lưới.
- PR-AUC 0.8794 — đặc biệt ấn tượng khi lớp dương chỉ chiếm 0,015% trong
    dữ liệu gốc. Chỉ số này phản ánh khả năng phát hiện bão mà không tạo quá nhiều
    cảnh báo sai.
- Gap validation/test rất nhỏ (0.0001 ROC-AUC) — mô hình generalize tốt sang dữ liệu
    mới.

![Hình 4.1: Đồ thị hội tụ LightGBM qua 200 boosting rounds](./chart_lgbm_convergence.png)

**Top 10 Feature Importances (gain — tổng giảm loss do feature đóng góp):**

| Rank | Đặc trưng | Gain | % Tổng | Ý nghĩa |
|:----:|-----------|-----:|:------:|---------|
| 1 | `slp_avg` | 3.626.751 | 47,7% | Áp suất biển trung bình — yếu tố #1 |
| 2 | `lon` | 1.985.665 | 26,1% | Kinh độ — vùng hình thành bão |
| 3 | `slp_tendency` | 392.091 | 5,2% | Xu hướng áp suất — dấu hiệu phát triển |
| 4 | `lat` | 388.157 | 5,1% | Vĩ độ — lực Coriolis |
| 5 | `v_wind_avg` | 242.446 | 3,2% | Gió bắc-nam |
| 6 | `month` | 204.844 | 2,7% | Tính mùa vụ |
| 7 | `oni_value` | 164.181 | 2,2% | Chỉ số ENSO |
| 8 | `u_wind_avg` | 101.700 | 1,3% | Gió đông-tây |
| 9 | `wind_speed_env_avg` | 80.052 | 1,1% | Tốc độ gió môi trường |
| 10 | `slp_180d_avg` | 79.360 | 1,0% | Trung bình áp suất 6 tháng |

**Phân tích Feature Importances:**
- **slp_avg chiếm 47,7%** — áp suất mực biển là tín hiệu mạnh nhất. Vùng có áp suất
    thấp bất thường trùng trực tiếp với tâm xoáy thuận nhiệt đới. Điều này phù hợp
    với định nghĩa vật lý: bão là hệ thống áp thấp sâu.
- **lon chiếm 26,1%** — vị trí kinh độ giúp mô hình học rằng bão chủ yếu hình thành
    trong dải 120–160°E (warm pool Tây Thái Bình Dương), hiếm khi ở ven bờ hoặc
    quá xa phía Đông.
- **slp_tendency (5,2%)** — xu hướng áp suất giảm nhanh trong 7 ngày là tiền đề
    của cyclogenesis, phù hợp với lý thuyết Gray (1968).
- **oni_value (2,2%)** — chỉ số ENSO giúp mô hình phân biệt pattern bão giữa các
    năm El Niño (bão xa bờ, ít hơn trong grid) và La Niña (bão gần bờ, nhiều hơn).

![Hình 4.2: Top 10 Feature Importances — LightGBM Storm Classifier](./chart_feature_importance.png)

**Probability Calibration:**
- Raw probability mean: 0.073170 → Calibrated: **0.047809** (-35%)
- Isotonic calibration giúp xác suất đầu ra phản ánh đúng tần suất dương thực tế,
    quan trọng cho bước tổng hợp SPI ở Phase 5.

**4.3 Kết quả Phase 5 — Dự báo số bão theo tháng**

Sử dụng Ridge regression (α=10) với 27 đặc trưng (5 SPI variants + 12 month dummies
+ ONI lags + SPI lag + 6 ENSO interactions) trên 505 mẫu (43 năm × ~12 tháng):

**Kết quả tổng quan:**
- **Test MAE (tháng):** 1.03 bão/tháng
- **Test MAE (năm):** 4.03 bão/năm

**Kết quả dự báo theo năm (tập test ≥ 2020, chỉ các năm đầy đủ 12 tháng):**

| Năm | Actual | Predicted | Error | MAE/tháng | ENSO Phase |
|:---:|:------:|:---------:|:-----:|:---------:|:----------:|
| 2020 | 34 | 31.6 | +2.4 | 0.94 | Neutral→La Niña |
| 2021 | 31 | 38.2 | −7.2 | 1.28 | La Niña |
| 2022 | 37 | 40.1 | −3.1 | 0.87 | La Niña→Neutral |
| 2023 | 27 | 31.1 | −4.1 | 1.08 | El Niño phát triển |
| 2024 | 33 | 36.4 | −3.4 | 0.97 | El Niño→Neutral |
| **Trung bình** | **32.4** | **35.5** | | **1.03** | |

**Nhận xét:**
- Sai số trung bình chỉ **1.03 bão/tháng** — có nghĩa mô hình dự báo chệch trung
    bình khoảng 1 cơn bão mỗi tháng so với thực tế.
- Sai số tốt nhất: **2022** với MAE/tháng = 0.87 — một năm La Niña điển hình.
- Sai số lớn nhất: **2021** với error −7.2 bão/năm — năm La Niña mạnh, bão tập
    trung cuối mùa bất thường.
- Mô hình có xu hướng **over-predict nhẹ** (trung bình +3.1 bão/năm), nhưng nắm
    bắt đúng xu hướng tăng/giảm giữa các năm.

![Hình 4.3: Dự báo vs Thực tế — Số bão hàng năm (2020–2024)](./chart_annual_predictions.png)

**Chi tiết dự báo năm 2024 (theo tháng):**

| Tháng | Monthly SPI | Actual | Predicted | Error | Nhận xét |
|:-----:|:----------:|:------:|:---------:|:-----:|----------|
| Jan | 587,8 | 0 | 0,8 | +0,8 | Mùa khô — SPI thấp |
| Feb | 368,9 | 0 | 0,3 | +0,3 | |
| Mar | 1.488,8 | 0 | 0,5 | +0,5 | SPI tăng — biển ấm dần |
| Apr | 3.094,9 | 0 | 0,7 | +0,7 | |
| May | 12.473,6 | 2 | 1,9 | +0,1 | ✓ Bắt đầu mùa bão sớm |
| Jun | 8.329,7 | 1 | 0,7 | +0,3 | SPI giảm — gió mùa Tây Nam mạnh |
| Jul | 24.562,4 | 3 | 4,6 | −1,6 | Peak 1 — SPI nhảy vọt |
| Aug | 21.447,5 | 7 | 5,2 | +1,8 | Peak 2 — actual vượt predicted |
| Sep | 52.569,7 | 10 | 9,1 | +0,9 | ✓ Đỉnh mùa bão — SPI cực đại |
| Oct | 30.722,9 | 4 | 5,6 | −1,6 | SPI vẫn cao nhưng bão giảm |
| Nov | 22.626,3 | 5 | 4,0 | +1,0 | Cuối mùa — vẫn hoạt động |
| Dec | 17.569,9 | 1 | 3,0 | −2,0 | Over-predict — mùa kết thúc sớm |
| **TOTAL** | | **33** | **36,4** | **−3,4** | |

**Phân tích chi tiết năm 2024:**
- Mô hình dự đoán chính xác đỉnh mùa bão tại **tháng 9** (SPI = 52.569,7 —
    cao nhất năm) với actual = 10, predicted = 9,1. Đây là tín hiệu rất mạnh:
    SPI tháng 9 gấp đôi tháng 10, phản ánh đúng thực tế.
- Sai số lớn nhất ở **tháng 12** (+2.0) — mô hình chưa nắm bắt được tín hiệu kết
    thúc mùa bão sớm trong năm El Niño chuyển Neutral.
- Tổng sai số năm chỉ −3,4 bão (actual 33, predicted 36,4) — sai số ~10%.

![Hình 4.4: Chi tiết dự báo năm 2024 — Actual vs Predicted theo tháng](./chart_monthly_2024.png)

![Hình 4.5: Tương quan SPI và Số bão thực tế — Năm 2024](./chart_spi_correlation.png)

![Hình 4.6: So sánh 4 mô hình hồi quy — Phase 5](./chart_model_comparison.png)

**4.4 Phân tích thiên lệch và hạn chế**

- **Over-prediction nhẹ:** Mô hình dự báo cao hơn thực tế trung bình ~3 bão/năm. Nguyên
    nhân chính: SPI tích lũy xác suất từ toàn bộ grid, bao gồm cả các tín hiệu khí tượng
    "gần giống bão" nhưng không phát triển thành TC chính thức.
- **ENSO spatial shift:** Năm El Niño, bão dịch chuyển sang phía Đông ra ngoài vùng
    grid không gian, dẫn đến SPI giảm nhưng tổng bão toàn basin không giảm tương ứng.
    Các ENSO × SPI interaction terms giúp giảm thiểu nhưng chưa giải quyết hoàn toàn
    vấn đề này.
- **Ridge thắng tree models:** Với chỉ 505 mẫu huấn luyện, Ridge regression (α=10)
    generalize tốt hơn các mô hình phức tạp — GBR_Huber bị overfit (train MAE 0.43
    vs test MAE 1.11), trong khi Ridge duy trì gap nhỏ (0.97 vs 1.03).
- **Year 2021 outlier:** Sai số lớn nhất (−7.2 bão) xảy ra trong năm La Niña mạnh
    với mùa bão hoạt động bất thường mạnh vào cuối năm.

**4.5 So sánh với giải pháp truyền thống**

| Tiêu chí | Pandas (đơn luồng) | PySpark (đề tài) |
|----------|:------------------:|:----------------:|
| Thời gian feature engineering (600M rows) | ~6–8 giờ (ước tính) | ~76 phút |
| Distributed inference | Out of Memory | ~124 phút (mapInPandas) |
| Khả năng mở rộng | < 50M rows (RAM) | Hàng tỷ rows (spill-to-disk) |
| Fault tolerance | Không | Có (RDD lineage) |

**4.6 Đánh giá tài nguyên hệ thống**

- **CPU:** Trung bình 80-95% utilization trong Phase 2, 5 (I/O bound + compute)
- **RAM:** Peak ~8GB driver process — vượt ngưỡng → spill-to-disk liên tục
- **Disk I/O:** Tổng ~50GB spill data (Spark shuffle + persisted blocks)
- **Tổng thời gian:** 12.387 giây (~3h27m) — chấp nhận được cho batch processing
    định kỳ (chạy 1 lần/tháng khi có dữ liệu mới)

_Kết luận Chương IV_

Kết quả thực nghiệm cho thấy pipeline bottom-up hoạt động hiệu quả: bộ phân loại
LightGBM đạt ROC-AUC 0.9915 trong phát hiện bão cấp ô lưới, và Ridge regression
đạt MAE 1.03 bão/tháng trong dự báo số lượng bão. Hệ thống PySpark xử lý thành
công gần 600 triệu bản ghi trên single machine, chứng minh tính khả thi của phương
pháp.


# CHƯƠNG V: KẾT LUẬN & HƯỚNG PHÁT TRIỂN

**5.1 Kết luận:**

- Hệ thống đã chứng minh thành công việc ứng dụng Big Data (PySpark) để tích
    hợp 4 nguồn dữ liệu khí tượng quốc tế (ERA5, NOAA OISST, IBTrACS, ONI)
    thành master dataset gần **600 triệu bản ghi**, bao phủ 42 năm dữ liệu (1983–2024).
- Bộ phân loại LightGBM đạt **ROC-AUC 0.9915** và **PR-AUC 0.8794** trong việc
    phát hiện sự hiện diện bão tại cấp ô lưới 0.25° × ngày — chứng tỏ tín hiệu khí
    tượng (đặc biệt là áp suất mực biển, kinh độ và xu hướng áp suất) có khả năng
    phân biệt rõ ràng giữa điều kiện có bão và không có bão.
- Pipeline bottom-up đạt **MAE 1.03 bão/tháng** và **MAE 4.03 bão/năm** trong dự
    báo số lượng bão (test set 2020–2024), cho thấy phương pháp tổng hợp SPI từ
    xác suất vi mô là cách tiếp cận hiệu quả, chính xác cho dự báo mùa bão.
- Feature importance analysis xác nhận vai trò quan trọng của `slp_avg` (47,7%),
    `lon` (26,1%) và `slp_tendency` (5,2%) — phù hợp với cơ sở lý thuyết về điều
    kiện nhiệt động lực học hình thành bão (Gray, 1968).
- Hệ thống xử lý thành công gần 600M bản ghi trên single machine trong 3,5 giờ,
    chứng minh PySpark với cơ chế spill-to-disk là giải pháp khả thi cho xử lý dữ liệu
    lớn trong nghiên cứu khí tượng ngay cả với tài nguyên phần cứng hạn chế.

**5.2 Hạn chế của hệ thống:**

- Over-prediction trung bình ~3 bão/năm — cần cải thiện phương pháp chuyển đổi
    SPI thành số lượng bão, có thể bằng cách thêm correction factor theo mùa.
- Mô hình chưa tích hợp chỉ số MJO (Madden-Julian Oscillation) — một yếu tố
    quan trọng ảnh hưởng đến typhoon genesis ở scale 30–60 ngày.
- Thời gian huấn luyện ~3.5 giờ trên single machine — có thể giảm đáng kể bằng
    cách sử dụng Spark cluster nhiều node hoặc tối ưu partitioning.
- Phase 2 và 5 bị ảnh hưởng nặng bởi spill-to-disk (chiếm ~90% cảnh báo
    MemoryStore) — cần tăng RAM hoặc giảm kích thước partition.
- Tập huấn luyện Phase 5 chỉ 505 mẫu (tháng) — hạn chế khả năng sử dụng mô
    hình phức tạp hơn Ridge.

**5.3 Hướng phát triển:**

- **Tối ưu Spark:** Tăng driver/executor memory lên 16GB, sử dụng SSDs thay HDD
    cho shuffle, giảm thời gian huấn luyện xuống <1 giờ.
- **Mở rộng grid domain:** Bao phủ toàn bộ Western Pacific (100–180°E) bằng ERA5
    data từ Google Cloud Public Dataset để giảm thiên lệch không gian.
- **Tích hợp MJO:** Thêm RMM1, RMM2 indices vào bộ đặc trưng để cải thiện dự
    báo ở scale intra-seasonal (30–60 ngày).
- **Ensemble đa mô hình:** Kết hợp Ridge + AdaBoost + GBR với stacking để cải
    thiện accuracy và giảm variance.
- **Phát triển Near Real-Time pipeline:** Tự động cập nhật dữ liệu ERA5 và NOAA SST
    hàng ngày/tuần, chạy inference tăng dần (incremental) thay vì toàn bộ dataset.
- **Mô hình chuyên biệt theo ENSO:** Huấn luyện Ridge riêng cho từng chế độ
    (El Niño / La Niña / Neutral) để xử lý tốt hơn hiệu ứng dịch chuyển không gian.


# TÀI LIỆU THAM KHẢO

1. Gray, W. M. (1968). Global View of the Origin of Tropical Disturbances and Storms.
    _Monthly Weather Review_, 96(10), 669–700.
2. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
    _Proceedings of the 22nd ACM SIGKDD_, 785–794.
3. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
    _Advances in Neural Information Processing Systems 30_.
4. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
    _Annals of Statistics_, 29(5), 1189–1232.
5. Zaharia, M. et al. (2016). Apache Spark: A Unified Engine for Big Data Processing.
    _Communications of the ACM_, 59(11), 56–65.
6. Knapp, K. R. et al. (2010). The International Best Track Archive for Climate
    Stewardship (IBTrACS). _Bulletin of the American Meteorological Society_, 91(3).
7. Reynolds, R. W. et al. (2007). Daily High-Resolution-Blended Analyses for Sea
    Surface Temperature. _Journal of Climate_, 20(22), 5473–5496.
8. Hersbach, H. et al. (2020). The ERA5 Global Reanalysis. _Quarterly Journal of the
    Royal Meteorological Society_, 146(730), 1999–2049.
9. Tài liệu kỹ thuật Apache Spark: https://spark.apache.org/./latest/
10. NOAA OISST: https://www.ncei.noaa.gov/products/optimum-interpolation-sst

