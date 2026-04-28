#!/usr/bin/env python
"""
timeseries.py

Developer: Cameron Cummins
Contact: cameron.cummins@utexas.edu
Last Header Update: 01/31/25
"""
import numpy as np
import fnmatch
from os.path import isfile
from os import remove, makedirs
from pathlib import Path
from gents.meta import get_attributes
from gents.mhfdataset import MHFDataset
from gents.datastore import GenTSDataStore
from gents.utils import get_version, LOG_LEVEL_IO_WARNING, ProgressBar
from gents.real_info_processor import RealInfoProcessor
import logging
from typing import Any

import sys
#sys.path.insert(1, '../real-information/src')
#import real_info

logger = logging.getLogger(__name__)

def check_timeseries_integrity(ts_path: str):
    """
    Checks whether a time-series file was written completely by GenTS.

    Opens the file and looks for the ``gents_version`` global attribute, which
    is stamped on every successfully completed output file.

    :param ts_path: Path to the time-series netCDF file to inspect.
    :type ts_path: str
    :returns: ``True`` if ``gents_version`` is present (file likely complete),
        ``False`` if absent or the file cannot be opened (possible corruption).
    :rtype: bool
    """
    try:
        with GenTSDataStore(ts_path, mode="r") as ts_ds:
            attrs = get_attributes(ts_ds)
        if "gents_version" in attrs:
            return True
    except OSError:
        logger.log(LOG_LEVEL_IO_WARNING, f"Corrupt timeseries output: '{ts_path}'")
    return False


def generate_time_series(hf_paths, ts_path_template, primary_var, secondary_vars, complevel=0, compression=None, overwrite=False, reference_structure=None, real_info_processor=None):
    """
    Checks whether a time-series file meets the GenTS chunking conventions.

    :param hf_paths: List of paths to history files to generate time series from
    :param ts_out_dir: Directory to output time series files to.
    :param prefix: Prefix to add to beginning of the names of the generated time series files.
    :param complevel: Compression level to apply through netCDF4 API.
    :param compression: Compression algorithm to use through netCDF4 API.
    :param overwrite: Whether or not to delete existing time series files with the same names as those being generated.
    :param target_variable: Primary variable to extract from history files.
    :param real_info_config_path: Path to real information config yaml file. 
    :return: List of paths to time series generated.
    """
    with GenTSDataStore(ts_path, mode="r") as ts_ds:
        if list(ts_ds["time"].chunking()) != list(ts_ds["time"].shape):
            return False
        for variable in ts_ds.variables:
            if list(ts_ds[variable].chunking()) == list(ts_ds[variable].shape):
                continue
                
            if "time" not in ts_ds[variable].dimensions or len(ts_ds[variable].shape) == 1:
                return False
            else:
                chunking = list(ts_ds[variable].chunking())
                chunking[0] += 1
                bumped_size = np.prod(chunking)*ts_ds[variable].dtype.itemsize
                if bumped_size < 4*(1024**2):
                    return False
        
    return True

    variables_list = ["T", "U", "V", "PRECC", "PRECL", "PSL", "TREFHT", "TREFHTMN"] #, "TREFHTMX", "TS", "FSNT", "FLNT", "FSNS", "FLNS"]
    ts_out_path = None
    with MHFDataset(hf_paths) as agg_hf_ds:
        secondary_vars_data = {}
        
        for variable in secondary_vars:
            secondary_vars_data[variable] = agg_hf_ds.get_var_vals(variable)
        
        for variable in ts_args:
            args = copy.deepcopy(ts_args[variable])
            ts_string = args["ts_string"]
            ts_out_path = f"{ts_path_template}.{variable}.{ts_string}.nc"
            del args["ts_string"]

            ts_paths.append(write_timeseries_file(
                agg_hf_ds=agg_hf_ds,
                ts_out_path=ts_out_path,
                primary_var=variable,
                secondary_vars_data=secondary_vars_data,
                **args
            ))
    
    return ts_paths


        bits_shaved = 0
        with netCDF4.Dataset(ts_out_path, mode="w") as ts_ds:
            #if primary_var is not None:
            if primary_var in variables_list:    # XXX: debugging
                var_shape = agg_hf_ds.get_var_data_shape(primary_var)
                var_dims = agg_hf_ds.get_var_dimensions(primary_var)
                for index, dim in enumerate(var_dims):
                    if dim == "time":
                        ts_ds.createDimension(dim, None)
                    else:
                        ts_ds.createDimension(dim, var_shape[index])

    :param dt: Duration of a single model time step.
    :type dt: datetime.timedelta
    :param subhour_format: Format string for sub-minute time steps. Defaults to ``'%Y%m%d%H%M%S'``.
    :type subhour_format: str
    :param hourly_format: Format string for hour-level time steps (< 24 h). Defaults to ``'%Y%m%d%H'``.
    :type hourly_format: str
    :param daily_format: Format string for day-level time steps (< 28 days). Defaults to ``'%Y%m%d'``.
    :type daily_format: str
    :param monthly_format: Format string for month-level time steps (< 12 months). Defaults to ``'%Y%m'``.
    :type monthly_format: str
    :param yearly_format: Format string for year-level time steps. Defaults to ``'%Y'``.
    :type yearly_format: str
    :returns: ``strftime``-compatible format string.
    :rtype: str
    """
    minutes = dt.total_seconds() / 60
    hours = minutes / 60
    days = hours / 24
    months = days / 30

                time_chunk_size = 1
                bits_shaved = []
                if len(var_shape) > 0 and "time" in var_dims:
                    #for i in range(0, var_shape[0], time_chunk_size):
                    for i in range(0, min(var_shape[0], 10), time_chunk_size): # XXX: debugging
                        if i + time_chunk_size > var_shape[0]:
                            time_chunk_size = var_shape[0] - i
                        input_data = agg_hf_ds.get_var_vals(primary_var, time_index_start=i, time_index_end=i+time_chunk_size)
                        var_data[i:i + time_chunk_size], shaved = real_info_processor.shave_data(input_data, agg_hf_ds, primary_var, i, time_chunk_size)

                else:
                    input_data = agg_hf_ds.get_var_vals(primary_var)
                    var_data[:], bits_shaved = real_info_processor.shave_data(input_data, agg_hf_ds, primary_var)

                ts_ds[primary_var].setncattr("bits_shaved", np.asarray(bits_shaved, dtype=np.int32)) # Want to move this to a secondary variable rather than attribute.

            for secondary_var in secondary_vars_data:
                var_shape = agg_hf_ds.get_var_data_shape(secondary_var)
                var_dims = agg_hf_ds.get_var_dimensions(secondary_var)

                for index, dim in enumerate(var_dims):
                    if dim not in ts_ds.dimensions:
                        if dim == "time":
                            ts_ds.createDimension(dim, None)
                        else:
                            try:
                                ts_ds.createDimension(dim, var_shape[index])
                            except:    # XXX: for some reason some of the shapes are just integers in the short run that I did.
                                ts_ds.createDimension(dim, var_shape)
                
                svar_data = ts_ds.createVariable(secondary_var,
                                                agg_hf_ds.get_var_dtype(secondary_var),
                                                var_dims,
                                                complevel=complevel,
                                                compression=compression)
                
                svar_data.set_auto_mask(False)
                svar_data.set_auto_scale(False)
                svar_data.set_always_mask(False)

                ts_ds[secondary_var].setncatts(agg_hf_ds.get_var_attrs(secondary_var))
                input_data = secondary_vars_data[secondary_var]
                svar_data[:], bits_shaved = real_info_processor.shave_data(input_data, agg_hf_ds, secondary_var, 0)
                ts_ds[secondary_var].setncattr("bits_shaved", np.int32(bits_shaved))
            
            ts_ds.setncatts(global_attrs | {"gents_version": str(get_version())})
    return ts_out_path


class TSCollection:
    """Time Series Collection that faciliates the creation of time series from a HFCollection."""
    def __init__(self, hf_collection, output_dir, ts_orders=None, dask_client=None, real_info_config_path=None):
        """
        :param hf_collection: History file collection to derive time series from
        :param output_dir: Directory to output time series files to
        :param ts_orders: List of Dask delayed functions of generate_time_series
        :param dask_client: Dask client to use when executing time series batches (Default: global client).
        :param real_info_config_path: Path to YAML configuration file containing real info settings
        """
        if dask_client is not None:
            warnings.warn("Dask is no longer implemented in GenTS. Use the 'num_processes' argument to enable parallelism or reference the ReadTheDocs for using Dask.", DeprecationWarning, stacklevel=2)

        self.__num_processes = 1
        if num_processes is not None:
            self.__num_processes = num_processes
        
        # Initialize the RealInfoProcessor with the provided parameters
        #print(f"ts collection initializing with real_info_config_path: {real_info_config_path}")
        self.__real_info_config_path = real_info_config_path
        self.__real_info_processor = RealInfoProcessor(config_path=real_info_config_path)
        #print(f"ts collection real info processor initialized with real_info_flag: {self.__real_info_processor.real_info_flag} and real_info_tol: {self.__real_info_processor.real_info_tol}")
        
        hf_collection.sort_along_time()

        self.__hf_collection = hf_collection
        self.__groups = self.__hf_collection.get_groups()
        self.__output_dir = output_dir
        
        if ts_orders is None:
            self.__hf_collection.pull_metadata()
            self.__orders = []
            for glob_template in self.__groups:
                output_template = glob_template.split(str(self.__hf_collection.get_input_dir()))[1]
                ts_path_template = f"{self.__output_dir}{output_template}"
                hf_paths = self.__groups[glob_template]

                primary_vars = self.__hf_collection[hf_paths[0]].get_primary_variables()
                secondary_vars = self.__hf_collection[hf_paths[0]].get_secondary_variables()

                if len(primary_vars) > 0:
                    for var in primary_vars:
                        self.__orders.append({
                            "hf_paths": hf_paths,
                            "ts_path_template": ts_path_template[:-1],
                            "primary_var": var,
                            "secondary_vars": secondary_vars#,
                            #"real_info_processor": self.__real_info_processor
                        })
                else:
                    self.__orders.append({
                        "hf_paths": hf_paths,
                        "ts_path_template": ts_path_template[:-1],
                        "primary_var": None,
                        "secondary_vars": secondary_vars#,
                        #"real_info_processor": self.__real_info_processor
                    })

            logger.debug(f"TSCollection initialized at '{output_dir}'.")
            logger.debug(f"{len(self.__orders)} timeseries orders generated.")
        else:
            self.__orders = ts_orders

    def __contains__(self, key):
        return key in self.__orders

    def __iter__(self):
        return iter(self.__orders)

    def __getitem__(self, index):
        return self.__orders[index]

    def __len__(self):
        return len(self.__orders)

    def items(self):
        return self.__orders

    def values(self):
        return self.__orders
    
    def get_hf_collection(self):
        """
        Returns the underlying ``HFCollection``.

        :returns: The history file collection this ``TSCollection`` was derived from.
        :rtype: gents.hfcollection.HFCollection
        """
        return self.__hf_collection
    
    def get_output_dir(self):
        """
        Returns the output directory path for generated time series files.

        :returns: Absolute path to the output directory.
        :rtype: str
        """
        return self.__output_dir

    def update_ts_orders(self, strfrmt_kwargs={}, time_alignment_method="midpoint"):
        """
        Rebuilds the time-series order list and returns a new ``TSCollection``.

        Re-derives one order per primary variable per history file group, applying
        ``strfrmt_kwargs`` to override individual timestamp format strings and
        ``time_alignment_method`` to control which point within each time bound is
        used when computing ``start_time`` / ``end_time`` for the output filename.

        Time alignment methods:

        - ``'midpoint'`` *(default)*: midpoint of the first time bound.
        - ``'direct_time'``: raw ``time`` coordinate values (ignores bounds).
        - ``'start_bound'``: lower edge of the first time bound.
        - ``'end_bound'``: upper edge of the first time bound.

        :param strfrmt_kwargs: Format-string overrides forwarded to
            :func:`get_timestamp_format` (e.g. ``{'monthly_format': '%Y%m%d'}``).
            Defaults to ``{}``.
        :type strfrmt_kwargs: dict
        :param time_alignment_method: Method used to select the representative
            time value from each file's time bounds. Must be one of
            ``'midpoint'``, ``'direct_time'``, ``'start_bound'``, or
            ``'end_bound'``. Defaults to ``'midpoint'``.
        :type time_alignment_method: str
        :returns: A new ``TSCollection`` with the rebuilt order list.
        :rtype: TSCollection
        :raises ValueError: If ``time_alignment_method`` is not one of the
            accepted values.
        """
        self.__hf_collection.check_pulled()
        orders = []
        for glob_template in self.__groups:
            output_template = glob_template.split(str(self.__hf_collection.get_input_dir()))[1]
            if "[sorting_pivot]" in output_template:
                output_template = output_template.split("[sorting_pivot]")[0]
            ts_path_template = f"{self.__output_dir}{output_template}"
            hf_paths = self.__groups[glob_template]

            primary_vars = self.__hf_collection[hf_paths[0]].get_primary_variables()
            secondary_vars = self.__hf_collection[hf_paths[0]].get_secondary_variables()
            time_format = get_timestamp_format(self.__hf_collection.get_timestep_delta(hf_paths[0]), **strfrmt_kwargs)
            
            times = []
            for path in hf_paths:
                time_bnds = self.__hf_collection[path].get_cftime_bounds()
                if time_alignment_method == "direct_time" or time_bnds is None:
                    time = self.__hf_collection[path].get_cftimes()
                elif time_alignment_method == "midpoint":
                    time = [time_bnds[0][0] + (time_bnds[0][1] - time_bnds[0][0]) / 2]
                elif time_alignment_method == "start_bound":
                    time = [time_bnds[0][0]]
                elif time_alignment_method == "end_bound":
                    time = [time_bnds[0][1]]
                else:
                    raise ValueError(f"'{time_alignment_method}' is an invalid time-alignment method. Valid methods are ['direct_time', 'midpoint', 'start_bound', 'end_bound']")

                times.append(time)
            times = np.concatenate(times)
            start_time = min(times)
            end_time = max(times)

            timestamp_str = f"{start_time.strftime(time_format)}-{end_time.strftime(time_format)}"

            if len(primary_vars) > 0:
                for var in primary_vars:
                    orders.append({
                        "hf_paths": hf_paths,
                        "ts_path_template": ts_path_template[:-1],
                        "primary_var": var,
                        "secondary_vars": secondary_vars,
                        "ts_string": timestamp_str
                    })
            else:
                orders.append({
                    "hf_paths": hf_paths,
                    "ts_path_template": ts_path_template[:-1],
                    "primary_var": "auxiliary",
                    "secondary_vars": secondary_vars,
                    "ts_string": timestamp_str
                })
        return self.copy(ts_orders=orders)

    def copy(self, hf_collection=None, output_dir=None, ts_orders=None, num_processes=None):
        """
        Creates a new ``TSCollection`` derived from this one with optional overrides.

        Used as the return mechanism for all modifier methods to preserve immutability.

        :param hf_collection: ``HFCollection`` to assign to the copy. Defaults to
            the current collection.
        :type hf_collection: gents.hfcollection.HFCollection or None
        :param output_dir: Output directory to assign to the copy. Defaults to the
            current directory.
        :type output_dir: str or None
        :param ts_orders: Order list to assign to the copy. Defaults to the current
            orders.
        :type ts_orders: list or None
        :param num_processes: Worker process count for the copy. Defaults to the
            current value.
        :type num_processes: int or None
        :returns: New ``TSCollection`` instance.
        :rtype: TSCollection
        """
        if hf_collection is None:
            hf_collection = self.__hf_collection
        if output_dir is None:
            output_dir = self.__output_dir
        if ts_orders is None:
            ts_orders = self.__orders
        if num_processes is None:
            num_processes = self.__num_processes

        return TSCollection(hf_collection=hf_collection, output_dir=output_dir, ts_orders=ts_orders, dask_client=dask_client, real_info_config_path=self.__real_info_config_path)

    def include(self, path_glob, var_glob="*"):
        """
        Returns a new collection containing only orders that match both filters.

        An order is retained if at least one of its source paths matches
        ``path_glob`` *and* its primary variable matches ``var_glob``.

        :param path_glob: ``fnmatch`` glob applied to source history file paths.
        :type path_glob: str
        :param var_glob: ``fnmatch`` glob applied to primary variable names.
            Defaults to ``'*'``.
        :type var_glob: str
        :returns: New ``TSCollection`` restricted to matching orders.
        :rtype: TSCollection
        """
        filtered_orders = []
        for order_dict in copy.deepcopy(self.__orders):
            path_matched = False
            for path in order_dict["hf_paths"]:
                if fnmatch.fnmatch(path, path_glob):
                    path_matched = True
                    break
            
            if path_matched and fnmatch.fnmatch(order_dict["primary_var"], var_glob):
                filtered_orders.append(order_dict)
        logger.debug(f"Inclusive filter(s) applied: '{var_glob}' to history files matching '{path_glob}'")
        return self.copy(ts_orders=filtered_orders)

    def exclude(self, path_glob, var_glob=""):
        """
        Returns a new collection with orders that match both filters removed.

        An order is excluded if any of its source paths matches ``path_glob`` *and*
        its primary variable matches ``var_glob``.

        :param path_glob: ``fnmatch`` glob applied to source history file paths.
        :type path_glob: str
        :param var_glob: ``fnmatch`` glob applied to primary variable names.
            Defaults to ``''``.
        :type var_glob: str
        :returns: New ``TSCollection`` with matching orders removed.
        :rtype: TSCollection
        """
        filtered_orders = []
        for order_dict in copy.deepcopy(self.__orders):
            path_unmatched = True
            for path in order_dict["hf_paths"]:
                if fnmatch.fnmatch(path, path_glob):
                    path_unmatched = False
                    break
            
            if path_unmatched and not fnmatch.fnmatch(order_dict["primary_var"], var_glob):
                filtered_orders.append(order_dict)
        logger.debug(f"Exclusive filter(s) applied: '{var_glob}' to history files matching '{path_glob}'")
        return self.copy(ts_orders=filtered_orders)

    def add_args(self, path_glob="*", var_glob="*", level=None, alg=None, overwrite=None):
        """
        Updates generation arguments on orders that match both filters.

        Only arguments that are not ``None`` are applied; others are left unchanged.

        :param path_glob: ``fnmatch`` glob applied to source history file paths.
            Defaults to ``'*'``.
        :type path_glob: str
        :param var_glob: ``fnmatch`` glob applied to primary variable names.
            Defaults to ``'*'``.
        :type var_glob: str
        :param level: netCDF4 compression level (0–9). Defaults to ``None``
            (unchanged).
        :type level: int or None
        :param alg: netCDF4 compression algorithm (e.g. ``'zlib'``). Defaults to
            ``None`` (unchanged).
        :type alg: str or None
        :param overwrite: Overwrite flag to apply. Defaults to ``None`` (unchanged).
        :type overwrite: bool or None
        :returns: New ``TSCollection`` with updated order arguments.
        :rtype: TSCollection
        """
        new_orders = []
        for order_dict in copy.deepcopy(self.__orders):
            path_matched = False
            for path in order_dict["hf_paths"]:
                if fnmatch.fnmatch(path, path_glob):
                    path_matched = True
                    break
            
            #print(f"order_dict: {order_dict} var_glob: {var_glob}")
            #if path_matched and fnmatch.fnmatch(order_dict["primary_var"], var_glob):
            if path_matched:
                if (order_dict["primary_var"] is not None):
                    if fnmatch.fnmatch(order_dict["primary_var"], var_glob):
                        if level is not None:
                            order_dict["complevel"] = level
                        if alg is not None:
                            order_dict["compression"] = alg
                        if overwrite is not None:
                            order_dict["overwrite"] = overwrite
            new_orders.append(order_dict)

        logger.debug(f"Arguments applied (excluding None): ['level': {level}, 'alg': {alg}, 'overwrite': {overwrite}] to history files matching '{path_glob}' and variables matching '{var_glob}'.")
        return self.copy(ts_orders=new_orders)

    def apply_path_swap(self, string_match, string_swap, path_glob="*", var_glob="*"):
        """
        Replaces a substring in the output path template of matching orders.

        Iterates over orders whose source paths match ``path_glob`` and replaces
        ``string_match`` with ``string_swap`` in each order's ``ts_path_template``.
        Used to redirect outputs to a different directory structure (e.g.
        ``'/hist/'`` → ``'/proc/tseries/'``).

        :param string_match: Substring to find in the output path template.
        :type string_match: str
        :param string_swap: Replacement string.
        :type string_swap: str
        :param path_glob: ``fnmatch`` glob applied to source history file paths.
            Defaults to ``'*'``.
        :type path_glob: str
        :param var_glob: ``fnmatch`` glob applied to primary variable names.
            Defaults to ``'*'``.
        :type var_glob: str
        :returns: New ``TSCollection`` with updated path templates.
        :rtype: TSCollection
        """
        new_orders = []
        for order_dict in copy.deepcopy(self.__orders):
            for path in order_dict["hf_paths"]:
                if fnmatch.fnmatch(path, path_glob):
                    order_dict["ts_path_template"] = order_dict["ts_path_template"].replace(string_match, string_swap)
            new_orders.append(order_dict)
    
        logger.debug(f"Path swap '{string_match}' -> '{string_swap}' to history files matching '{path_glob}' and variables matching '{var_glob}'.")
        return self.copy(ts_orders=new_orders)
        
    def apply_compression(self, level, alg, path_glob, var_glob="*"):
        """
        Applies compression settings to matching time-series orders.

        Convenience wrapper around :meth:`add_args`.

        :param level: netCDF4 compression level (0–9).
        :type level: int
        :param alg: netCDF4 compression algorithm (e.g. ``'zlib'``).
        :type alg: str
        :param path_glob: ``fnmatch`` glob applied to source history file paths.
        :type path_glob: str
        :param var_glob: ``fnmatch`` glob applied to primary variable names.
            Defaults to ``'*'``.
        :type var_glob: str
        :returns: New ``TSCollection`` with compression arguments applied.
        :rtype: TSCollection
        """
        return self.add_args(path_glob=path_glob, var_glob=var_glob, level=level, alg=alg)

    def apply_overwrite(self, path_glob, var_glob="*"):
        """
        Sets the overwrite flag on matching time-series orders.

        Convenience wrapper around :meth:`add_args` with ``overwrite=True``.

        :param path_glob: ``fnmatch`` glob applied to source history file paths.
        :type path_glob: str
        :param var_glob: ``fnmatch`` glob applied to primary variable names.
            Defaults to ``'*'``.
        :type var_glob: str
        :returns: New ``TSCollection`` with overwrite enabled on matching orders.
        :rtype: TSCollection
        """
        return self.add_args(path_glob=path_glob, var_glob=var_glob, overwrite=True)

    def append_timestep_dirs(self, var_glob="*"):
        """
        Inserts a time-step frequency subdirectory into each matching order's output path.

        Determines the frequency label from the group's timestep delta:
        ``'hour_N'``, ``'day_N'``, ``'month_N'``, or ``'year_N'``.  The label is
        inserted as a new directory level immediately before the filename in the
        output path template, organising outputs by observation frequency.

        :param var_glob: ``fnmatch`` glob applied to primary variable names.
            Defaults to ``'*'``.
        :type var_glob: str
        :returns: New ``TSCollection`` with updated output path templates.
        :rtype: TSCollection
        """
        new_orders = []
        for order_dict in copy.deepcopy(self.__orders):
            if fnmatch.fnmatch(order_dict["primary_var"], var_glob):
                dt = self.__hf_collection.get_timestep_delta(order_dict["hf_paths"][0])

                if dt is None:
                    timestep_label = "unsorted"
                else:
                    hours = np.rint(dt.total_seconds() / 60.0 / 60.0)
                    days = np.rint(hours / 24.0)
                    months = np.rint(days / 30)
                    years = np.rint(months / 12)
                    if hours < 24:
                        timestep_label = f"hour_{int(hours)}"
                    elif days < 28:
                        timestep_label = f"day_{int(days)}"
                    elif months < 12:
                        timestep_label = f"month_{int(months)}"
                    else:
                        timestep_label = f"year_{int(years)}"

                template = Path(order_dict["ts_path_template"])
                order_dict["ts_path_template"] = str(template.parent) + f"/{timestep_label}/" + template.name

                new_orders.append(order_dict)
        return self.copy(ts_orders=new_orders)

    def remove_overwrite(self, path_glob, var_glob="*"):
        """
        Clears the overwrite flag on matching time-series orders.

        Convenience wrapper around :meth:`add_args` with ``overwrite=False``.

        :param path_glob: ``fnmatch`` glob applied to source history file paths.
        :type path_glob: str
        :param var_glob: ``fnmatch`` glob applied to primary variable names.
            Defaults to ``'*'``.
        :type var_glob: str
        :returns: New ``TSCollection`` with overwrite disabled on matching orders.
        :rtype: TSCollection
        """
        return self.add_args(path_glob=path_glob, var_glob=var_glob, overwrite=False)

    def create_directories(self, exist_ok=True):
        """
        Creates the output directory tree for all time-series orders.

        :param exist_ok: If ``True`` (default), no error is raised when a
            directory already exists.
        :type exist_ok: bool
        """
        logger.info("Creating directory structure for time series output.")
        for order_dict in self.__orders:
            makedirs(Path(order_dict['ts_path_template']).parent, exist_ok=exist_ok)

    def execute(self, optimize=True, optimize_batch_n=200, raise_errors=False):
        """
        Executes all time-series generation orders in parallel.

        When ``optimize=True`` (default), orders that share the same first source
        file are batched together (up to ``optimize_batch_n`` per batch) so that
        :func:`generate_time_series` opens each group of history files only once
        and writes multiple primary-variable output files per worker invocation,
        significantly reducing file I/O overhead.

        When ``optimize=False``, each order is submitted as a separate worker task
        (one file open per variable).

        :param optimize: If ``True`` (default), batch orders sharing the same
            source files into single worker calls.
        :type optimize: bool
        :param optimize_batch_n: Maximum number of variables per optimised batch.
            Defaults to ``200``.
        :type optimize_batch_n: int
        :param raise_errors: If ``True`` (default ``False``), calls errors are raised
            rather than just logged.
        :type raise_errors: bool
        :returns: List of paths to all generated time-series output files.
        :rtype: list[str]
        """
        self.create_directories()
        results = []
        if self.__dask_client is None:
            logger.info("No Dask client detected... proceeding in serial.")
            prog_bar = ProgressBar(total=len(self.__orders), write_progress=False)
            for args in self.__orders:
                args["real_info_processor"] = self.__real_info_processor
                results.append(generate_time_series(**args))
                prog_bar.step()
        else:
            for index, order in enumerate(self.__orders):
                args = copy.deepcopy(order)
                del args["hf_paths"]
                del args["ts_path_template"]
                del args["secondary_vars"]
                del args["primary_var"]
                ts_args = {order["primary_var"]: args}
                optimized_orders.append({
                    "hf_paths": order["hf_paths"],
                    "ts_path_template": order["ts_path_template"],
                    "secondary_vars": order["secondary_vars"],
                    "ts_args": ts_args
                })

        with ProcessPoolExecutor(max_workers=self.__num_processes) as executor:
            futures = {executor.submit(generate_time_series, **args): args for args in optimized_orders}
            prog_bar = ProgressBar(total=len(futures), label="Generating Timeseries")
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    path = futures[future]
                    logger.warning(f"Failed to load metadata for {path}: {exc}", exc_info=True)
                    if raise_errors:
                        raise
                finally:
                    prog_bar.step()
        
        output_paths = []
        for result in results:
            for path in result:
                output_paths.append(path)

        return output_paths