function final_table = database_select_range_of_data(base_path, topo_range, tech_nodes_range, vdd_range, vcm_range, cl_range, run_range)
    % base_path 是数据库的基础路径
    % topo_range, tech_nodes_range, ... 是各层次的选定范围，可以是具体值的数组或包含 '*' 表示所有值
    % 初始化最终的表格
    final_table = [];
    % 定义一个内部函数来检查范围
    function match = in_range(value, range)
        if ischar(range) && strcmp(range, '*')
            match = true;
        else
            match = any(strcmp(value, range));
        end
    end

    % 获取 base_path 下所有层级的文件夹
    topos = dir(fullfile(base_path, '*'));
    topos = topos([topos.isdir] & ~ismember({topos.name}, {'.', '..'}));

    % 遍历所有 Topo 文件夹
    for i = 1:length(topos)
        if in_range(topos(i).name, topo_range)
            tech_nodes_path = fullfile(base_path, topos(i).name);
            tech_nodes = dir(fullfile(tech_nodes_path, '*'));
            tech_nodes = tech_nodes([tech_nodes.isdir] & ~ismember({tech_nodes.name}, {'.', '..'}));

            % 遍历所有 Tech_nodes 文件夹
            for j = 1:length(tech_nodes)
                if in_range(tech_nodes(j).name, tech_nodes_range)
                    vdd_path = fullfile(tech_nodes_path, tech_nodes(j).name);
                    vdds = dir(fullfile(vdd_path, '*'));
                    vdds = vdds([vdds.isdir] & ~ismember({vdds.name}, {'.', '..'}));

                    % 遍历所有 VDD 文件夹
                    for k = 1:length(vdds)
                        if in_range(vdds(k).name, vdd_range)
                            vcm_path = fullfile(vdd_path, vdds(k).name);
                            vcms = dir(fullfile(vcm_path, '*'));
                            vcms = vcms([vcms.isdir] & ~ismember({vcms.name}, {'.', '..'}));

                            % 遍历所有 VCM 文件夹
                            for l = 1:length(vcms)
                                if in_range(vcms(l).name, vcm_range)
                                    cl_path = fullfile(vcm_path, vcms(l).name);
                                    cls = dir(fullfile(cl_path, '*'));
                                    cls = cls([cls.isdir] & ~ismember({cls.name}, {'.', '..'}));

                                    % 遍历所有 CL 文件夹
                                    for m = 1:length(cls)
                                        if in_range(cls(m).name, cl_range)
                                            run_path = fullfile(cl_path, cls(m).name);
                                            runs = dir(fullfile(run_path, '*'));
                                            runs = runs([runs.isdir] & ~ismember({runs.name}, {'.', '..'}));

                                            % 遍历所有 RUN 文件夹
                                            for n = 1:length(runs)
                                                if in_range(runs(n).name, run_range)
                                                    final_path = fullfile(run_path, runs(n).name, 'ext_mat', 'Perf_pop_data.mat');

                                                    % 检查文件是否存在
                                                    if exist(final_path, 'file')
                                                        loaded_data = load(final_path);
                                                        % 检查数据是否有 PerfTable
                                                        if isfield(loaded_data, 'newTable')
                                                            % 拆分路径并为每部分添加列
                                                            path_parts = {topos(i).name, tech_nodes(j).name, vdds(k).name, vcms(l).name, cls(m).name, runs(n).name};
                                                            loaded_data.newTable.Topo = repmat(path_parts(1), height(loaded_data.newTable), 1);
                                                            loaded_data.newTable.Tech_nodes = str2double(repmat(path_parts(2), height(loaded_data.newTable), 1));
                                                            loaded_data.newTable.VDD = str2double(repmat(path_parts(3), height(loaded_data.newTable), 1));
                                                            loaded_data.newTable.VCM = str2double(repmat(path_parts(4), height(loaded_data.newTable), 1));
                                                            loaded_data.newTable.CL = str2double(repmat(path_parts(5), height(loaded_data.newTable), 1));
                                                            loaded_data.newTable.RUN = str2double(repmat(path_parts(6), height(loaded_data.newTable), 1));

                                                            if isempty(final_table)
                                                                final_table = loaded_data.newTable;
                                                            else
                                                                final_table = [final_table; loaded_data.newTable]; %#ok<AGROW>
                                                            end
                                                            disp(['Loaded data from ', final_path]);
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
