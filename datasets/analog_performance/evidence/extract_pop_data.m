function extract_pop_data(src_path, extraction_mat_loc)
    % 设置文件模式
    file_pattern = fullfile(src_path, 'pop_gen*.txt');

    % 获取所有匹配的文件
    files = dir(file_pattern);

    % 初始化表格变量
    T = table();

    % 遍历每个文件
    for f = 1:length(files)
        file_name = files(f).name;
        full_path = fullfile(files(f).folder, file_name);

        % 打开文件并读取内容
        fid = fopen(full_path, 'r');
        file_content = fread(fid, '*char')';
        fclose(fid);

        % 提取gen的值
        gen = str2double(regexp(file_name, '(?<=pop_gen)\d+', 'match', 'once'));

        % 提取每一个index的数据
        index_data = regexp(file_content, 'inid_index(?<index>\d+):\s*(?<data>\{.*?\})', 'names');

        for k = 1:length(index_data)
                current_index = str2double(index_data(k).index);
                data_str = index_data(k).data;

                % 转换数据字符串为标准的JSON格式
                data_str_corrected = strrep(data_str, '''', '"');
                data_str_corrected = regexprep(data_str_corrected, '(\w+):', '"$1":');
                try
                    data_struct = jsondecode(data_str_corrected);
                catch
                    warning('Failed to decode JSON for index %d in file %s', current_index, file_name);
                    continue;
                end
                fields = fieldnames(data_struct);

                for i = 1:length(fields)
                    if ischar(data_struct.(fields{i}))
                        % 尝试转换为数字
                        num_val = str2double(data_struct.(fields{i}));
                        if ~isnan(num_val)
                            data_struct.(fields{i}) = num_val;
                        end
                    end
                end


                data_table = struct2table(data_struct);

                if(data_table.pm<0)
                    continue
                end
                % 添加gen和index列
                data_table.gen = gen;
                data_table.index = current_index;

                 % 处理chip_area列的数据, 取其绝对值并开平方
                if ismember('chip_area', fields)
                    data_table.chip_area = sqrt(abs(data_table.chip_area));
                end
                % 得到FOM
                data_table.FOM_AW = computeFOM(data_table.gbw,data_table.PSRR, data_table.CMRR, data_table.gain, data_table.ivdd_27, data_table.tc, data_table.noise, data_table.vos);

                % 将当前数据追加到总表格
                T = [T; data_table];
        end
    end

    T = delete_duplicate_rows(T);
    % 根据文件名生成表格变量名
    table_var_name = matlab.lang.makeValidName('pop_data');

    % 提取 TBM 表
    extractAndSaveTable(T, '^(ft|gm|VDS|gmid_offset|PVT_offset)\d+$', 'TBM', extraction_mat_loc, table_var_name);

    % 提取性能表
    selectedColumns = {'gen', 'index', 'pm', 'gbw', 'gain', 'noise', 'CMRR', 'PSRR', 'SR', 'ivdd_27', 'vos', 'tc', 'd_settle', 'settlingTime','FOMS','FOML','chip_area', 'fitness', 'FOM_AW'};
    newTable = T(:, selectedColumns);
    newTableName = ['Perf_' table_var_name];
    filename = strcat(extraction_mat_loc,'\', newTableName, '.mat');
    save(filename, 'newTable');

    % 提取 TD 表
    extractAndSaveTable(T, '^(MOSFET_|CURRENT_|RESISTOR_|CAPACITOR_)\w+$', 'TD', extraction_mat_loc, table_var_name);

    disp('数据提取完成！');
end
