function extract_pop_data(src_path, extraction_mat_loc)
    % Set the filename pattern.
    file_pattern = fullfile(src_path, 'pop_gen*.txt');

    % Find all matching files.
    files = dir(file_pattern);

    % Initialize the result table.
    T = table();

    % Process each file.
    for f = 1:length(files)
        file_name = files(f).name;
        full_path = fullfile(files(f).folder, file_name);

        % Open the file and read its contents.
        fid = fopen(full_path, 'r');
        file_content = fread(fid, '*char')';
        fclose(fid);

        % Extract the generation number.
        gen = str2double(regexp(file_name, '(?<=pop_gen)\d+', 'match', 'once'));

        % Extract data for each individual index.
        index_data = regexp(file_content, 'inid_index(?<index>\d+):\s*(?<data>\{.*?\})', 'names');

        for k = 1:length(index_data)
                current_index = str2double(index_data(k).index);
                data_str = index_data(k).data;

                % Convert the data string to standard JSON syntax.
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
                        % Attempt conversion to a numeric value.
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
                % Add the gen and index columns.
                data_table.gen = gen;
                data_table.index = current_index;

                 % Transform chip_area by taking the square root of its absolute value.
                if ismember('chip_area', fields)
                    data_table.chip_area = sqrt(abs(data_table.chip_area));
                end
                % Compute the figure of merit.
                data_table.FOM_AW = computeFOM(data_table.gbw,data_table.PSRR, data_table.CMRR, data_table.gain, data_table.ivdd_27, data_table.tc, data_table.noise, data_table.vos);

                % Append the current data to the result table.
                T = [T; data_table];
        end
    end

    T = delete_duplicate_rows(T);
    % Derive the table variable name from the filename.
    table_var_name = matlab.lang.makeValidName('pop_data');

    % Extract the TBM table.
    extractAndSaveTable(T, '^(ft|gm|VDS|gmid_offset|PVT_offset)\d+$', 'TBM', extraction_mat_loc, table_var_name);

    % Extract the performance table.
    selectedColumns = {'gen', 'index', 'pm', 'gbw', 'gain', 'noise', 'CMRR', 'PSRR', 'SR', 'ivdd_27', 'vos', 'tc', 'd_settle', 'settlingTime','FOMS','FOML','chip_area', 'fitness', 'FOM_AW'};
    newTable = T(:, selectedColumns);
    newTableName = ['Perf_' table_var_name];
    filename = strcat(extraction_mat_loc,'\', newTableName, '.mat');
    save(filename, 'newTable');

    % Extract the TD table.
    extractAndSaveTable(T, '^(MOSFET_|CURRENT_|RESISTOR_|CAPACITOR_)\w+$', 'TD', extraction_mat_loc, table_var_name);

    disp('Data extraction complete.');
end
