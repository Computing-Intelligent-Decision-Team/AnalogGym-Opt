function final_table = database_select_range_of_data(base_path, topo_range, tech_nodes_range, vdd_range, vcm_range, cl_range, run_range)
    % base_path is the root directory of the database.
    % topo_range, tech_nodes_range, etc. select values at each level; '*' selects all values.
    % Initialize the combined result table.
    final_table = [];
    % Define a helper to check the selected range.
    function match = in_range(value, range)
        if ischar(range) && strcmp(range, '*')
            match = true;
        else
            match = any(strcmp(value, range));
        end
    end

    % List directories below base_path.
    topos = dir(fullfile(base_path, '*'));
    topos = topos([topos.isdir] & ~ismember({topos.name}, {'.', '..'}));

    % Iterate over topology directories.
    for i = 1:length(topos)
        if in_range(topos(i).name, topo_range)
            tech_nodes_path = fullfile(base_path, topos(i).name);
            tech_nodes = dir(fullfile(tech_nodes_path, '*'));
            tech_nodes = tech_nodes([tech_nodes.isdir] & ~ismember({tech_nodes.name}, {'.', '..'}));

            % Iterate over technology-node directories.
            for j = 1:length(tech_nodes)
                if in_range(tech_nodes(j).name, tech_nodes_range)
                    vdd_path = fullfile(tech_nodes_path, tech_nodes(j).name);
                    vdds = dir(fullfile(vdd_path, '*'));
                    vdds = vdds([vdds.isdir] & ~ismember({vdds.name}, {'.', '..'}));

                    % Iterate over VDD directories.
                    for k = 1:length(vdds)
                        if in_range(vdds(k).name, vdd_range)
                            vcm_path = fullfile(vdd_path, vdds(k).name);
                            vcms = dir(fullfile(vcm_path, '*'));
                            vcms = vcms([vcms.isdir] & ~ismember({vcms.name}, {'.', '..'}));

                            % Iterate over VCM directories.
                            for l = 1:length(vcms)
                                if in_range(vcms(l).name, vcm_range)
                                    cl_path = fullfile(vcm_path, vcms(l).name);
                                    cls = dir(fullfile(cl_path, '*'));
                                    cls = cls([cls.isdir] & ~ismember({cls.name}, {'.', '..'}));

                                    % Iterate over CL directories.
                                    for m = 1:length(cls)
                                        if in_range(cls(m).name, cl_range)
                                            run_path = fullfile(cl_path, cls(m).name);
                                            runs = dir(fullfile(run_path, '*'));
                                            runs = runs([runs.isdir] & ~ismember({runs.name}, {'.', '..'}));

                                            % Iterate over run directories.
                                            for n = 1:length(runs)
                                                if in_range(runs(n).name, run_range)
                                                    final_path = fullfile(run_path, runs(n).name, 'ext_mat', 'Perf_pop_data.mat');

                                                    % Check whether the file exists.
                                                    if exist(final_path, 'file')
                                                        loaded_data = load(final_path);
                                                        % Check for the performance table.
                                                        if isfield(loaded_data, 'newTable')
                                                            % Split the path and add a column for each component.
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
